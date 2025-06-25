#!/bin/bash

# O script coleta a porcentagem de espaço em disco usando o comando df
# compara com um limite predefinido (90% neste exemplo) e emite um alerta se exceder.

limite=90
espaco=$(df -h | awk 'NR==2 {print $5}' | sed 's/%//')

if [ $espaco -gt $limite ]; then
  echo "Alerta: Espaço em disco excedeu $limite%."
else
  echo "Espaço em disco está abaixo do limite."
fi
