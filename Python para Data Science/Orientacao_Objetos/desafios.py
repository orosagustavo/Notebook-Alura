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

'''
- Crie uma nova classe chamada Pessoa com atributos como nome, idade e profissão.
- Adicione um método especial __str__ para imprimir uma representação em string da pessoa.
- Implemente também um método de instância chamado aniversario que aumenta a idade da 
pessoa em um ano.
- Por fim, adicione uma propriedade chamada saudacao que retorna uma mensagem de 
saudação personalizada com base na profissão da pessoa.
'''
class Pessoa:
    def __init__(self, nome, idade, profissao = ''):
        self.nome = nome.title()
        self.idade = idade
        self.profissao = profissao
        
    def __str__(self):
        return f'{self.nome}, {self.idade} anos, {self.profissao}'
    
    def aniversario(self):
        self.idade += 1
    
    def saudacao(self):
        if self.profissao:
            print(f'Sou {self.nome}, trabalho como {self.profissao}')
        else:
            print(f'Sou {self.nome}')

jose = Pessoa('jose',25,'Barbeiro')
miguel = Pessoa('Miguel',12)

miguel.saudacao()