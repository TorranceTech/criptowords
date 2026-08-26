#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <CL/cl.h>
#include "pbkdf2_hmac512.hpp"

namespace gpu {

class Engine {
public:
    Engine() : ctx_(nullptr), cmdq_(nullptr), initialized_(false), is_gpu_(false),
               kernel_(nullptr), program_(nullptr) {}
    ~Engine() { cleanup(); }

    bool init(const std::string& kernel_file = "") {
        cl_platform_id plats[4];
        cl_uint np = 0;
        if (clGetPlatformIDs(4, plats, &np) != CL_SUCCESS || np == 0) return false;

        cl_device_id best_dev = nullptr;
        is_gpu_ = false;

        for (cl_uint p = 0; p < np; p++) {
            cl_device_id devs[4];
            cl_uint nd = 0;
            if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 4, devs, &nd) != CL_SUCCESS) continue;
            for (cl_uint d = 0; d < nd; d++) {
                cl_device_type dt;
                clGetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(dt), &dt, nullptr);
                if (dt == CL_DEVICE_TYPE_GPU || dt == CL_DEVICE_TYPE_ACCELERATOR) {
                    best_dev = devs[d]; is_gpu_ = true;
                }
                if (!best_dev) best_dev = devs[d];
            }
        }
        if (!best_dev) return false;

        char name[256] = {};
        clGetDeviceInfo(best_dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        device_name_ = name;

        cl_int err;
        ctx_ = clCreateContext(nullptr, 1, &best_dev, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) return false;

        cmdq_ = clCreateCommandQueue(ctx_, best_dev, 0, &err);
        if (err != CL_SUCCESS) { cleanup(); return false; }

        // Compilar kernel se fornecido
        if (!kernel_file.empty()) {
            if (!build_kernel(kernel_file)) {
                std::cerr << "[GPU] Falha ao compilar kernel: " << kernel_file << std::endl;
                // Nao falha a inicializacao — pode usar CPU
            }
        }

        initialized_ = true;
        return true;
    }

    // =================================================================
    // pbkdf2_batch: processa batch de PBKDF2-HMAC-SHA512 na GPU
    // =================================================================
    bool pbkdf2_batch(
        const std::vector<std::string>& mnemonics,
        int iterations,
        std::vector<std::vector<uint8_t>>& seeds)
    {
        if (!initialized_ || !kernel_) return false;

        size_t n = mnemonics.size();
        seeds.resize(n, std::vector<uint8_t>(64, 0));

        // Preparar senhas como pares (ptr, len)
        std::vector<std::pair<const uint8_t*, size_t>> passwords;
        passwords.reserve(n);
        for (const auto& m : mnemonics) {
            passwords.emplace_back(reinterpret_cast<const uint8_t*>(m.data()), m.size());
        }

        // Preparar buffers para o kernel
        static const uint8_t salt[] = "mnemonic";
        auto kbuf = pbkdf2::prepare_for_kernel(passwords, salt, 8);

        // Criar buffers OpenCL
        cl_int err;
        cl_mem d_ipad = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        kbuf.ipad_states.size(), kbuf.ipad_states.data(), &err);
        if (err != CL_SUCCESS) return false;

        cl_mem d_opad = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        kbuf.opad_states.size(), kbuf.opad_states.data(), &err);
        if (err != CL_SUCCESS) { clReleaseMemObject(d_ipad); return false; }

        cl_mem d_salt = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        kbuf.salt_padded.size(), kbuf.salt_padded.data(), &err);
        if (err != CL_SUCCESS) { clReleaseMemObject(d_ipad); clReleaseMemObject(d_opad); return false; }

        cl_mem d_out = clCreateBuffer(ctx_, CL_MEM_WRITE_ONLY,
                                       n * 64, nullptr, &err);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(d_ipad); clReleaseMemObject(d_opad); clReleaseMemObject(d_salt);
            return false;
        }

        // Definir argumentos do kernel
        uint32_t salt_padded_len = static_cast<uint32_t>(kbuf.salt_padded_len);
        uint32_t salt_blocks = static_cast<uint32_t>(kbuf.salt_blocks);
        uint32_t iter = static_cast<uint32_t>(iterations);
        uint32_t dk_len = 64;

        err  = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &d_ipad);
        err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &d_opad);
        err |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &d_salt);
        err |= clSetKernelArg(kernel_, 3, sizeof(uint32_t), &salt_padded_len);
        err |= clSetKernelArg(kernel_, 4, sizeof(uint32_t), &salt_blocks);
        err |= clSetKernelArg(kernel_, 5, sizeof(uint32_t), &iter);
        err |= clSetKernelArg(kernel_, 6, sizeof(uint32_t), &dk_len);
        err |= clSetKernelArg(kernel_, 7, sizeof(cl_mem), &d_out);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(d_ipad); clReleaseMemObject(d_opad);
            clReleaseMemObject(d_salt); clReleaseMemObject(d_out);
            return false;
        }

        // Executar
        size_t global_size = n;
        err = clEnqueueNDRangeKernel(cmdq_, kernel_, 1, nullptr,
                                      &global_size, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(d_ipad); clReleaseMemObject(d_opad);
            clReleaseMemObject(d_salt); clReleaseMemObject(d_out);
            return false;
        }

        // Ler resultados
        std::vector<uint8_t> results(n * 64);
        err = clEnqueueReadBuffer(cmdq_, d_out, CL_TRUE, 0, n * 64,
                                   results.data(), 0, nullptr, nullptr);

        // Liberar buffers
        clReleaseMemObject(d_ipad);
        clReleaseMemObject(d_opad);
        clReleaseMemObject(d_salt);
        clReleaseMemObject(d_out);

        if (err != CL_SUCCESS) return false;

        // Copiar para seeds
        for (size_t i = 0; i < n; i++) {
            std::memcpy(seeds[i].data(), results.data() + i * 64, 64);
        }

        return true;
    }

    bool is_initialized() const { return initialized_; }
    bool is_gpu() const { return is_gpu_; }
    const std::string& device_name() const { return device_name_; }

private:
    bool build_kernel(const std::string& kernel_file) {
        // Ler arquivo do kernel
        std::ifstream ifs(kernel_file);
        if (!ifs.is_open()) {
            // Tentar caminho relativo ao executavel
            std::cerr << "[GPU] Kernel file not found: " << kernel_file << std::endl;
            return false;
        }
        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string src = ss.str();
        const char* src_ptr = src.c_str();
        size_t src_len = src.size();

        cl_int err;
        program_ = clCreateProgramWithSource(ctx_, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS) return false;

        err = clBuildProgram(program_, 1, nullptr, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            // Log do erro de compilacao
            size_t log_size;
            clGetProgramBuildInfo(program_, nullptr, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program_, nullptr, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "[GPU] Build log:\n" << log.data() << std::endl;
            clReleaseProgram(program_);
            program_ = nullptr;
            return false;
        }

        kernel_ = clCreateKernel(program_, "pbkdf2_sha512", &err);
        if (err != CL_SUCCESS) {
            clReleaseProgram(program_);
            program_ = nullptr;
            return false;
        }

        return true;
    }

    void cleanup() {
        if (kernel_) clReleaseKernel(kernel_);
        if (program_) clReleaseProgram(program_);
        if (cmdq_) clReleaseCommandQueue(cmdq_);
        if (ctx_) clReleaseContext(ctx_);
        kernel_ = nullptr; program_ = nullptr;
        cmdq_ = nullptr; ctx_ = nullptr;
        initialized_ = false;
    }

    cl_context ctx_ = nullptr;
    cl_command_queue cmdq_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
    bool initialized_ = false;
    bool is_gpu_ = false;
    std::string device_name_;
};

} // namespace gpu
