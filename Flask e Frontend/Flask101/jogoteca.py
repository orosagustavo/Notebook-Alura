from flask import Flask, render_template, request

#Definir classe de jogos
class Jogo():
    def __init__(self, nome, console):
        self.nome = nome
        self.console = console

#Definição de uma nova lista
jogo1 = Jogo('God of War', 'PC')
jogo2 = Jogo('COD', 'PS5')
jogo3 = Jogo('Shadow of Colossus', 'PS2')
jogos = [jogo1, jogo2, jogo3]

#Armazena a aplicação em um objeto
app = Flask(__name__)

#Define uma rota para uma função
@app.route('/')
def index():
    return render_template('listaDinamica.html', titulo = 'Jogoteca',
                           jogos = jogos)

#Página com formulário para criar novo item
@app.route('/form')
def formulario():
    return render_template('form.html', titulo = 'Adicione um novo jogo')

@app.route('/adicionar', methods = ['POST',]) #Colocando POST
def adicionar():
    #Recebendo informações do formulário
    nome_jogo = request.form['nome']
    console_jogo = request.form['console']

    #Definindo novo jogo como objeto e colocando na lista
    novo_jogo = Jogo(nome=nome_jogo, console=console_jogo)
    jogos.append(novo_jogo)
    return render_template('listaDinamica.html', titulo = 'Jogos', jogos = jogos)

app.run(debug = True)