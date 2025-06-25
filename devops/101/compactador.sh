#! /bin/bash

if [ "$#" -lt 2 ]; then
	echo "O programa precisa do nome do arquivo e quais arquivos vão ser compactados"
	exit 1
fi

# Recebe o primeiro parâmetro do usuário
arquivo_saida=$1
arquivos=("${@:2}")

tar -czf "$arquivo_saida" "${arquivos[@]}"

echo "Os arquivos foram compactados. $arquivo_saida criado com sucesso"
