#!/usr/bin/env bash
# ============================================================
#  Busca dos SEUS dados. Edite as 2 linhas abaixo (M e H) e rode:
#      ./buscar.sh
# ============================================================
cd "$(dirname "$0")"

# 1) Seu mnemonico. Ponha ? onde nao sabe a palavra. Tudo entre aspas, numa linha so.
M="abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon ?"

# 2) O endereco-alvo do puzzle (comeca com 1).
H="1LqBGSKuX5yYUonjxT5qGfpUsXKYYWeabA"

# ------------------------------------------------------------
echo "Mnemonico: $M"
echo "Palavras : $(echo "$M" | wc -w)   (tem que dar 12 ou 24)"
echo "Alvo     : $H"
echo
./runner --mnemonic "$M" --hash "$H" --lang english --gpu
