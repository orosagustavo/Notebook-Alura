import requests
import json

# Puxando dados
url = 'https://guilhermeonrails.github.io/api-restaurantes/restaurantes.json'
response = requests.get(url)
dados = response.json()

# Filtrando dados
restaurantes = {}
for item in dados:
    nome_restaurante = item['Company']
    if nome_restaurante not in restaurantes:
        restaurantes[nome_restaurante] = []
    restaurantes[nome_restaurante].append({
        'item': item['Item'],
        'price': item['price'],
        'description': item['description'],
    })

#Criando os arquivos
for nome_do_restaurante, dados in restaurantes.items():
    nome_do_arquivo = f'{nome_do_restaurante}.json'
    with open(nome_do_arquivo,'w') as arquivo_restaurante:
        json.dump(dados,arquivo_restaurante,indent=4)