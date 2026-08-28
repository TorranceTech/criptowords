# criptowords

Recuperador de seed **BIP39** por força bruta: você informa o mnemônico com `?` nas
posições que não sabe e o endereço-alvo, e o programa descobre as palavras que faltam.
Usa CPU (multithread) e, com `--gpu`, **todas** as GPUs OpenCL detectadas em paralelo.

> Suporta dois tipos de alvo, detectados automaticamente pelo `--hash`:
> - **Bitcoin** — BIP44 legacy `m/44'/0'/0'/0/0` → endereços `1...`.
> - **Ethereum** — BIP44 `m/44'/60'/0'/0/0` + Keccak-256 → endereços `0x...`.
>
> (Endereços BTC `3...` SegWit-P2SH ou `bc1...` bech32 **não** são suportados no momento.)

---

## Novidades desta versão

O que foi adicionado/corrigido em relação à versão anterior:

- **Derivação BIP44 correta (`m/44'/0'/0'/0/0`).** Agora deriva a chave pela árvore
  BIP32 (CKD) e gera o endereço legacy `1...` **igual ao que as carteiras reais mostram**.
  Antes o endereço saía da chave-mestra `m` direto, sem caminho de derivação — não
  correspondia a nenhuma carteira. Validado contra uma referência independente:
  `abandon`×11 + `about` → `1LqBGSKuX5yYUonjxT5qGfpUsXKYYWeabA`.
- **Suporte a Ethereum.** Detecta automaticamente alvos `0x...` e usa BIP44
  `m/44'/60'/0'/0/0` + **Keccak-256** (comparação ignora maiúsc./minúsc. do EIP-55).
  Validado: a seed de teste → `0x9858EfFD232B4033E47d90003D41EC34EcaEda94`.
- **Suporte a múltiplas GPUs.** Com `--gpu`, o programa detecta e usa **todas** as GPUs
  OpenCL ao mesmo tempo (testado com RTX 3060 + GTX 1070).
- **Correções no motor OpenCL** (`gpu_engine.hpp`): o `clBuildProgram` recebia uma lista
  de device nula (erro `CL_INVALID_VALUE`), e o tratamento de erro estourava com
  `std::bad_alloc` — o kernel nunca compilava. Corrigido; agora roda de fato na GPU.
- **Pipeline paralelo** (`brute_engine.cpp`): enquanto as GPUs calculam o próximo lote,
  a CPU verifica o anterior (double-buffering), com a verificação BIP44 distribuída em
  várias threads. Resultado: ~61 mil tentativas/s usando as duas placas.
- **Build corrigido:** `-std=c++23` (o `c++26` do comando antigo não existe no g++ 13) e
  `-DCL_TARGET_OPENCL_VERSION=300`. Requer `libsecp256k1-dev`.
- **Facilidades:** scripts `src/teste.sh` (mostra um `MATCH FOUND` de exemplo) e
  `src/buscar.sh` (você edita o mnemônico e o alvo e roda), além deste README e um
  `.gitignore` para o binário.

> **Ainda em aberto:** BTC só endereços `1...` (não `3...`/`bc1...`); não valida o checksum
> do BIP39 (testa também combinações inválidas).

---

## 1. Requisitos

- `libsecp256k1-dev`, `libssl-dev` (OpenSSL) e um runtime OpenCL (ex.: driver NVIDIA).

```bash
sudo apt install -y libsecp256k1-dev libssl-dev
```

## 2. Compilar

Entre na pasta `src/` e compile (gera o executável `runner` em `src/`):

```bash
cd src
```
```bash
g++ -O3 -march=native -mavx2 -std=c++23 -DCL_TARGET_OPENCL_VERSION=300 main.cpp cli.cpp brute_engine.cpp cli_parser.cpp -o runner -lsecp256k1 -lcrypto -lOpenCL -lpthread
```

> **Importante:** sempre rode o `runner` de dentro de `src/` — ele procura o kernel
> `pbkdf2_hmac512.cl` e as wordlists em `../wordlist/` por caminho relativo.

---

## 3. Testar

> **Regra de ouro da colagem:** o mnemônico inteiro fica **entre aspas, numa linha só,
> sem apertar Enter no meio**. Se quebrar a linha dentro das aspas, a quebra vira parte
> do mnemônico e o resultado dá errado. Confira a contagem com `echo "$M" | wc -w`
> (tem que dar **12** ou **24**).

### Teste 1 — sanidade (deriva 1 endereço)
Deve imprimir `BTC: 1LqBGSKuX5yYUonjxT5qGfpUsXKYYWeabA`:

```bash
./runner --mnemonic "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about" --lang english
```

### Teste 2 — força bruta na GPU (descobre a última palavra)
Aqui a última palavra é `?`. O programa deve achar `about` e imprimir `MATCH FOUND (GPU)!`.
As três linhas abaixo são curtas de propósito (não quebram na colagem); cole **uma de cada vez**:

```bash
M="$(printf 'abandon %.0s' {1..11})?"
```
```bash
H=1LqBGSKuX5yYUonjxT5qGfpUsXKYYWeabA
```
```bash
./runner --mnemonic "$M" --hash "$H" --lang english --gpu
```

- A 1ª linha monta os 11 `abandon` + `?` automaticamente (sem contar na mão).
- A 2ª guarda o endereço-alvo.
- A 3ª roda a busca.

### Teste 3 — velocidade (as duas GPUs sob carga)
Duas palavras desconhecidas (`? ?`) + alvo inexistente → varre tudo (~4,2M) e mostra a taxa:

```bash
M="$(printf 'abandon %.0s' {1..10})? ?"
```
```bash
./runner --mnemonic "$M" --hash 1AlvoInexistenteXXXXXXXXXXXXXXXXXX --lang english --gpu
```

Enquanto roda, em **outro terminal**, veja as duas placas trabalhando:

```bash
watch -n1 nvidia-smi
```

---

## 4. Uso real (seus puzzles)

Escreva o que você sabe e ponha `?` onde não sabe. Ex. (uma linha só, sem quebra):

```bash
M="palavra1 palavra2 ? palavra4 palavra5 ? palavra7 palavra8 palavra9 palavra10 palavra11 palavra12"
```
```bash
H=<endereço_do_puzzle_começando_com_1>
```
```bash
./runner --mnemonic "$M" --hash "$H" --lang english --gpu
```

Confira antes de rodar:

```bash
echo "$M" | wc -w    # tem que dar 12 ou 24
```

---

## 5. Opções

| Opção | Descrição |
|---|---|
| `--mnemonic "..."` | Palavras; use `?` nas posições desconhecidas |
| `--hash <endereço>` | Endereço alvo a encontrar — BTC `1...` ou ETH `0x...` (auto-detectado) |
| `--lang <idioma>` | Wordlist (padrão: `english`) |
| `--wordlist <arquivo>` | Wordlist customizada |
| `--fix "pos:palavra"` | Fixa uma palavra numa posição (posições começam em 0) |
| `--allow "pos:pal1\|pal2"` | Restringe as palavras possíveis de uma posição |
| `--threads <n>` | Nº de threads de verificação na CPU (padrão: todas) |
| `--rounds <n>` | Iterações PBKDF2 (padrão: 2048, BIP39) |
| `--gpu` | Usa todas as GPUs OpenCL. Sem isso, roda só na CPU |
| `--help` | Ajuda |

Variáveis de ambiente (opcionais): `GPU_DEBUG=1` mostra a divisão de trabalho por placa;
`GPU_BALANCE=1` liga o balanceamento experimental por velocidade de GPU (por padrão a
divisão é igual entre as placas, que se mostrou mais rápida e estável).

---

## Observações

- O programa **não valida o checksum** do BIP39 (a última palavra), então ele testa também
  combinações inválidas — funciona, mas não é o mais eficiente.
- Alvos são **auto-detectados**: `0x...` → Ethereum (`m/44'/60'/0'/0/0` + Keccak-256);
  senão → Bitcoin legacy (`m/44'/0'/0'/0/0`). ETH compara ignorando maiúsc./minúsc. (EIP-55).
- Por estar em teste, podem ocorrer erros — reporte via issues.
