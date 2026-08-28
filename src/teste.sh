#!/usr/bin/env bash
# Teste de match conhecido: deve terminar com "MATCH FOUND (GPU)!"
# Uso:  ./teste.sh
cd "$(dirname "$0")"

M="$(printf 'abandon %.0s' {1..11})?"
H=1LqBGSKuX5yYUonjxT5qGfpUsXKYYWeabA

./runner --mnemonic "$M" --hash "$H" --lang english --gpu
