from abc import ABC, abstractmethod

class ItemCardapio(ABC): #ABC pois estamos usando um método abstrato
    def __init__(self, nome, preco):
        self._nome = nome
        self._preco = preco
    
    @abstractmethod # Definimos que todos os herdeiros devem ter esse método implementado
    def aplicar_desconto(self):
        pass