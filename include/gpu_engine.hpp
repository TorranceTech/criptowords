#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <CL/cl.h>
#include "pbkdf2_hmac512.hpp"

namespace gpu {

// Um device OpenCL com seu proprio contexto/fila/kernel (contexto por-device
// para permitir execucao concorrente em multiplas GPUs).
struct DeviceCtx {
    cl_device_id dev = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    std::string name;
};

class Engine {
public:
    Engine() = default;
    ~Engine() { cleanup(); }

    // Enumera TODAS as GPUs/aceleradores disponiveis e compila o kernel em cada.
    bool init(const std::string& kernel_file = "") {
        // Ler fonte do kernel uma vez
        std::string src;
        if (!kernel_file.empty()) {
            std::ifstream ifs(kernel_file);
            if (!ifs.is_open()) {
                std::cerr << "[GPU] Kernel file not found: " << kernel_file << std::endl;
                return false;
            }
            std::stringstream ss;
            ss << ifs.rdbuf();
            src = ss.str();
        }

        cl_platform_id plats[8];
        cl_uint np = 0;
        if (clGetPlatformIDs(8, plats, &np) != CL_SUCCESS || np == 0) return false;

        // 1a passada: coletar GPUs/aceleradores. Se nao houver, cai para qualquer device.
        std::vector<cl_device_id> gpu_devs;
        std::vector<cl_device_id> any_devs;
        for (cl_uint p = 0; p < np; p++) {
            cl_device_id devs[16];
            cl_uint nd = 0;
            if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 16, devs, &nd) != CL_SUCCESS) continue;
            for (cl_uint d = 0; d < nd; d++) {
                cl_device_type dt;
                clGetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(dt), &dt, nullptr);
                any_devs.push_back(devs[d]);
                if (dt == CL_DEVICE_TYPE_GPU || dt == CL_DEVICE_TYPE_ACCELERATOR) {
                    gpu_devs.push_back(devs[d]);
                }
            }
        }

        std::vector<cl_device_id>& chosen = gpu_devs.empty() ? any_devs : gpu_devs;
        if (chosen.empty()) return false;
        is_gpu_ = !gpu_devs.empty();

        // Criar contexto/fila/kernel para cada device escolhido
        for (cl_device_id dev : chosen) {
            DeviceCtx dc;
            dc.dev = dev;

            char name[256] = {};
            clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            dc.name = name;

            cl_int err;
            dc.ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) continue;

            dc.queue = clCreateCommandQueue(dc.ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
            if (err != CL_SUCCESS) { clReleaseContext(dc.ctx); continue; }

            if (!src.empty()) {
                if (!build_kernel(dc, src)) {
                    std::cerr << "[GPU] Falha ao compilar kernel em: " << dc.name << std::endl;
                    if (dc.queue) clReleaseCommandQueue(dc.queue);
                    if (dc.ctx) clReleaseContext(dc.ctx);
                    continue;
                }
            }
            devices_.push_back(dc);
        }

        if (devices_.empty()) return false;
        rate_.assign(devices_.size(), 0.0);   // items/s por device (0 = ainda desconhecido)

        // Nome combinado (para exibicao)
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (i) device_name_ += " + ";
            device_name_ += devices_[i].name;
        }
        initialized_ = true;
        return true;
    }

    // =================================================================
    // pbkdf2_batch: distribui o batch entre TODAS as GPUs, em paralelo.
    // =================================================================
    bool pbkdf2_batch(
        const std::vector<std::string>& mnemonics,
        int iterations,
        std::vector<std::vector<uint8_t>>& seeds)
    {
        if (!initialized_ || devices_.empty()) return false;

        size_t n = mnemonics.size();
        seeds.assign(n, std::vector<uint8_t>(64, 0));
        if (n == 0) return true;

        std::vector<uint8_t> flat(n * 64);   // saida plana [n*64]
        size_t ndev = devices_.size();

        // --- 1. Pre-calcula os estados HMAC ipad/opad de TODO o batch, em paralelo.
        //     Antes isso era feito dentro da thread de cada GPU, penalizando a placa
        //     que recebia a fatia maior. Fora do caminho critico, a divisao por
        //     velocidade de GPU passa a valer de fato.
        std::vector<uint8_t> ipad(n * 64), opad(n * 64);
        {
            unsigned hw = std::thread::hardware_concurrency();
            int pt = static_cast<int>(std::min<size_t>(hw ? hw : 4, n));
            std::vector<std::thread> pts;
            pts.reserve(pt);
            for (int t = 0; t < pt; ++t) {
                pts.emplace_back([&, t, pt]() {
                    uint64_t ip[8], op[8];
                    for (size_t i = static_cast<size_t>(t); i < n; i += pt) {
                        const auto& m = mnemonics[i];
                        pbkdf2::compute_hmac_states(
                            reinterpret_cast<const uint8_t*>(m.data()), m.size(), ip, op);
                        uint8_t* ipb = ipad.data() + i * 64;
                        uint8_t* opb = opad.data() + i * 64;
                        for (int j = 0; j < 8; ++j)
                            for (int k = 0; k < 8; ++k) {
                                ipb[j * 8 + k] = static_cast<uint8_t>(ip[j] >> (56 - 8 * k));
                                opb[j * 8 + k] = static_cast<uint8_t>(op[j] >> (56 - 8 * k));
                            }
                    }
                });
            }
            for (auto& t : pts) t.join();
        }

        // Salt (compartilhado) preparado uma unica vez.
        static const uint8_t salt[] = "mnemonic";
        size_t salt_padded_len = 0, salt_blocks = 0;
        std::vector<uint8_t> salt_padded =
            pbkdf2::prepare_salt_padded(salt, 8, salt_padded_len, salt_blocks);

        // --- 2. Reparticao ponderada pela velocidade PURA de GPU de cada placa.
        //     Enquanto alguma taxa for desconhecida, divide igualmente (bootstrap).
        // Divisao igual por padrao (empiricamente mais rapida e estavel neste
        // hardware/kernel: a carga por-batch e dominada por overhead de CPU/launch,
        // nao pelo tempo puro de kernel). O balanceamento por velocidade de GPU e
        // opcional via GPU_BALANCE=1 para experimentacao.
        static const bool use_balance = std::getenv("GPU_BALANCE") != nullptr;
        bool have_all_rates = use_balance && std::all_of(rate_.begin(), rate_.end(),
                                          [](double r){ return r > 0.0; });
        std::vector<size_t> counts(ndev, 0), offs(ndev, 0);
        double W = 0.0;
        for (double r : rate_) W += (r > 0.0 ? r : 1.0);
        size_t assigned = 0;
        for (size_t di = 0; di < ndev; ++di) {
            size_t c;
            if (di == ndev - 1)      c = n - assigned;   // ultimo device leva o resto
            else if (have_all_rates) c = static_cast<size_t>((static_cast<double>(n) * rate_[di]) / W + 0.5);
            else                     c = (n + ndev - 1) / ndev;  // bootstrap: divisao igual
            if (c > n - assigned) c = n - assigned;
            offs[di] = assigned;
            counts[di] = c;
            assigned += c;
        }

        if (std::getenv("GPU_DEBUG")) {
            std::cerr << "[DBG] split:";
            for (size_t di = 0; di < ndev; ++di)
                std::cerr << " " << devices_[di].name << "=" << counts[di]
                          << "(rate=" << rate_[di] << ")";
            std::cerr << std::endl;
        }

        // --- 3. Cada GPU processa sua fatia (somente trabalho de GPU nesta etapa).
        std::atomic<bool> ok{true};
        std::vector<std::thread> ts;
        ts.reserve(ndev);
        for (size_t di = 0; di < ndev; ++di) {
            if (counts[di] == 0) continue;
            ts.emplace_back([&, di]() {
                double gpu_secs = 0.0;
                if (!run_on_device(devices_[di],
                                   ipad.data() + offs[di] * 64, opad.data() + offs[di] * 64,
                                   salt_padded, static_cast<uint32_t>(salt_padded_len),
                                   static_cast<uint32_t>(salt_blocks),
                                   counts[di], iterations,
                                   flat.data() + offs[di] * 64, &gpu_secs)) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                // Taxa baseada no tempo PURO de GPU (imune a contencao de CPU).
                if (gpu_secs > 0.0) {
                    double measured = static_cast<double>(counts[di]) / gpu_secs;
                    rate_[di] = (rate_[di] > 0.0) ? (0.5 * rate_[di] + 0.5 * measured)
                                                  : measured;
                }
            });
        }
        for (auto& t : ts) t.join();

        if (!ok.load(std::memory_order_relaxed)) return false;

        for (size_t i = 0; i < n; ++i)
            std::memcpy(seeds[i].data(), flat.data() + i * 64, 64);
        return true;
    }

    bool is_initialized() const { return initialized_; }
    bool is_gpu() const { return is_gpu_; }
    size_t device_count() const { return devices_.size(); }
    const std::string& device_name() const { return device_name_; }

private:
    // Executa PBKDF2 numa fatia de `count` itens usando estados HMAC ipad/opad
    // JA pre-computados (ipad_slice/opad_slice apontam para count*64 bytes).
    // Escreve count*64 bytes em out. Somente trabalho de GPU aqui.
    static bool run_on_device(DeviceCtx& d,
                              const uint8_t* ipad_slice, const uint8_t* opad_slice,
                              const std::vector<uint8_t>& salt_padded,
                              uint32_t salt_padded_len, uint32_t salt_blocks,
                              size_t count, int iterations, uint8_t* out,
                              double* out_gpu_secs = nullptr)
    {
        if (!d.kernel || count == 0) return false;

        cl_int err;
        cl_mem d_ipad = clCreateBuffer(d.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       count * 64, const_cast<uint8_t*>(ipad_slice), &err);
        if (err != CL_SUCCESS) return false;
        cl_mem d_opad = clCreateBuffer(d.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       count * 64, const_cast<uint8_t*>(opad_slice), &err);
        if (err != CL_SUCCESS) { clReleaseMemObject(d_ipad); return false; }
        cl_mem d_salt = clCreateBuffer(d.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       salt_padded.size(), const_cast<uint8_t*>(salt_padded.data()), &err);
        if (err != CL_SUCCESS) { clReleaseMemObject(d_ipad); clReleaseMemObject(d_opad); return false; }
        cl_mem d_out = clCreateBuffer(d.ctx, CL_MEM_WRITE_ONLY, count * 64, nullptr, &err);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(d_ipad); clReleaseMemObject(d_opad); clReleaseMemObject(d_salt);
            return false;
        }

        uint32_t iter = static_cast<uint32_t>(iterations);
        uint32_t dk_len = 64;

        err  = clSetKernelArg(d.kernel, 0, sizeof(cl_mem), &d_ipad);
        err |= clSetKernelArg(d.kernel, 1, sizeof(cl_mem), &d_opad);
        err |= clSetKernelArg(d.kernel, 2, sizeof(cl_mem), &d_salt);
        err |= clSetKernelArg(d.kernel, 3, sizeof(uint32_t), &salt_padded_len);
        err |= clSetKernelArg(d.kernel, 4, sizeof(uint32_t), &salt_blocks);
        err |= clSetKernelArg(d.kernel, 5, sizeof(uint32_t), &iter);
        err |= clSetKernelArg(d.kernel, 6, sizeof(uint32_t), &dk_len);
        err |= clSetKernelArg(d.kernel, 7, sizeof(cl_mem), &d_out);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(d_ipad); clReleaseMemObject(d_opad);
            clReleaseMemObject(d_salt); clReleaseMemObject(d_out);
            return false;
        }

        size_t global_size = count;
        cl_event kern_ev = nullptr;
        err = clEnqueueNDRangeKernel(d.queue, d.kernel, 1, nullptr,
                                     &global_size, nullptr, 0, nullptr, &kern_ev);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(d_ipad); clReleaseMemObject(d_opad);
            clReleaseMemObject(d_salt); clReleaseMemObject(d_out);
            return false;
        }

        err = clEnqueueReadBuffer(d.queue, d_out, CL_TRUE, 0, count * 64,
                                  out, 0, nullptr, nullptr);

        // Tempo PURO de execucao do kernel na GPU (imune a contencao de CPU),
        // usado para o balanceamento de carga entre as placas.
        if (out_gpu_secs && kern_ev) {
            cl_ulong ts = 0, te = 0;
            if (clGetEventProfilingInfo(kern_ev, CL_PROFILING_COMMAND_START, sizeof(ts), &ts, nullptr) == CL_SUCCESS &&
                clGetEventProfilingInfo(kern_ev, CL_PROFILING_COMMAND_END, sizeof(te), &te, nullptr) == CL_SUCCESS &&
                te > ts) {
                *out_gpu_secs = static_cast<double>(te - ts) * 1e-9;
            }
        }
        if (kern_ev) clReleaseEvent(kern_ev);

        clReleaseMemObject(d_ipad);
        clReleaseMemObject(d_opad);
        clReleaseMemObject(d_salt);
        clReleaseMemObject(d_out);

        return err == CL_SUCCESS;
    }

    bool build_kernel(DeviceCtx& d, const std::string& src) {
        const char* src_ptr = src.c_str();
        size_t src_len = src.size();

        cl_int err;
        d.program = clCreateProgramWithSource(d.ctx, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS) return false;

        err = clBuildProgram(d.program, 1, &d.dev, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(d.program, d.dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            if (log_size > 1) {
                std::vector<char> log(log_size + 1, '\0');
                clGetProgramBuildInfo(d.program, d.dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                std::cerr << "[GPU] Build log (" << d.name << "):\n" << log.data() << std::endl;
            } else {
                std::cerr << "[GPU] clBuildProgram falhou (err=" << err << ") em " << d.name << "\n";
            }
            clReleaseProgram(d.program);
            d.program = nullptr;
            return false;
        }

        d.kernel = clCreateKernel(d.program, "pbkdf2_sha512", &err);
        if (err != CL_SUCCESS) {
            clReleaseProgram(d.program);
            d.program = nullptr;
            return false;
        }
        return true;
    }

    void cleanup() {
        for (auto& d : devices_) {
            if (d.kernel) clReleaseKernel(d.kernel);
            if (d.program) clReleaseProgram(d.program);
            if (d.queue) clReleaseCommandQueue(d.queue);
            if (d.ctx) clReleaseContext(d.ctx);
        }
        devices_.clear();
        initialized_ = false;
    }

    std::vector<DeviceCtx> devices_;
    std::vector<double> rate_;   // throughput medido por device (items/s), 0 = desconhecido
    std::string device_name_;
    bool initialized_ = false;
    bool is_gpu_ = false;
};

} // namespace gpu
