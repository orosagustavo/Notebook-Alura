from modelos.avaliacao import Avaliacao
from modelos.cardapio.item_cardapio import ItemCardapio

class Restaurante:
    restaurantes = []
    def __init__(self, nome, categoria):
        self._nome = nome.title() # usando o _ pro usuário definir uma vez e não conseguir muidar
        self.categoria = categoria.upper()
        self._ativo = False
        self._avaliacao = []
        self._cardapio = []
        Restaurante.restaurantes.append(self) #Pega o objeto(self) e coloca na lista criada
    def __str__(self):
        return f'Objeto: {self._nome}'
    
    @classmethod
    def listar_restaurantes(cls):
        print(f'{"Nome".ljust(25)} | {"Categoria".ljust(25)} | {"Avaliação".ljust(25)} | {"Status"} ')
        for restaurante in Restaurante.restaurantes:
            print(f'{restaurante._nome.ljust(25)} | {restaurante.categoria.ljust(25)} | {str(restaurante.media_avaliacoes).ljust(25)} | {restaurante._ativo}')
    
    @property
    def ativo(self):
        return 'Verdadeiro' if self._ativo else 'False'
    
    def alternar_estado(self):
        self._ativo = not self._ativo

    def receber_avaliacao(self, cliente, nota):
        avaliacao = Avaliacao(cliente, nota)
        self._avaliacao.append(avaliacao)
    
    @property
    def media_avaliacoes(self):
        if not self._avaliacao:
            return 0
        soma = sum(avaliacao._nota for avaliacao in self._avaliacao) 
        quantidade = len(self._avaliacao)
        media = round(soma/quantidade,1)
        return media
    
    def adicionar_cardapio(self, prato):
        if isinstance(prato, ItemCardapio):
            self._cardapio.append(prato)

    @property
    def listar_cardapio(self):
        print(f'Cardapio do Restaurante {self._nome}\n')
        for i, item in enumerate(self._cardapio, start = 1):
            if hasattr(item, 'descricao'):
                mensagem = f'{i}. Nome: {item._nome} | Preço: R$ {item._preco} | Descrição: {item.descricao}'
                print(mensagem)
            if hasattr(item, 'tamanho'):
                mensagem = f'{i}. Nome: {item._nome} | Preço: R$ {item._preco} | Tamanho: {item.tamanho}'
                print(mensagem) 
