from modelos.classes import Restaurante

restaurante_praca = Restaurante('praça', 'Gourmet')
restaurante_mexicano = Restaurante('MexicanFood', 'Mexicana')

restaurante_mexicano.receber_avaliacao('Gui',4)
restaurante_mexicano.receber_avaliacao('Milena', 10)
restaurante_mexicano.receber_avaliacao('Carlos', 5)

restaurante_mexicano.alternar_estado()

def main():
    Restaurante.listar_restaurantes()

if __name__ == '__main__':
    main()