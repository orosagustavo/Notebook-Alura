'''
Implemente uma classe chamada Carro com os atributos básicos, como modelo, cor e ano. 
Crie uma instância dessa classe e atribua valores aos seus atributos.
'''
class Carro:
    def __init__(self, modelo, cor, ano):
        self.modelo = modelo
        self.cor = cor
        self.ano = ano

meu_carro = Carro('Fusca','Verde',1978)

'''
Crie uma classe chamada Restaurante com os atributos nome, categoria, ativo e crie mais 2 
atributos. Instancie um restaurante e atribua valores aos seus atributos.
'''
class Restaurante:
    def __init__(self, nome, categoria, estrelas, michelin):
        self.nome = nome
        self.categoria = categoria
        self.ativo = False
        self.estrelas = estrelas
        self.michelin = michelin

rango_bom = Restaurante(nome = 'Rango Bom', categoria='Caseiro', estrelas= 5, michelin= 2)

