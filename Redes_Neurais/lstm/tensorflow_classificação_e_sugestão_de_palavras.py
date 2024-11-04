# -*- coding: utf-8 -*-
"""TensorFlow - Classificação e Sugestão de Palavras.ipynb

## Pré-processamento
O problema que será tratado será uma classificação de artigos. O modelo deverá sugerir palavras assim como um corretor. 
"""

# URL dos dados de treinamento (arquivo compactado com dados)
url ='https://github.com/allanspadini/curso-tensorflow-proxima-palavra/raw/main/dados/train.zip'

import pandas as pd

# Carrega os dados em um DataFrame e define os nomes das colunas
dados = pd.read_csv(url, header=None, names=['ClassIndex', 'Título', 'Descrição'])
dados.head()

# Ajusta a variável de índice da classe para começar em zero
dados['ClassIndex'] = dados['ClassIndex'] - 1

# Cria uma nova coluna 'Texto' combinando 'Título' e 'Descrição'
dados['Texto'] = dados['Título'] + ' ' + dados['Descrição']

"""Separando os dados de treino e teste"""
from sklearn.model_selection import train_test_split

# Divisão dos dados em treino e teste (80% treino e 20% teste)
x_train, x_test, y_train, y_test = train_test_split(dados['Texto'].values,
                                                    dados['ClassIndex'].values,
                                                    test_size=0.2, random_state=4256)

"""Transformação do texto em um vetor utilizando tokenização"""
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, losses

# Definindo o tamanho do vocabulário
VOCAB_SIZE = 1000

# Cria o encoder para transformar texto em vetores de inteiros
encoder = layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(x_train)

"""## Rede Neural - Definição da Arquitetura"""
model = Sequential([
    encoder,
    # Camada de Embedding para representar o contexto entre as palavras
    layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=16, mask_zero=False),
    # Camada convolucional para extrair características dos textos
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(),
    layers.Conv1D(128, kernel_size=3, activation='relu'),
    # Pooling global para transformar a saída em uma forma adequada para camadas densas
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.4),
    # Camada densa com ativação 'relu' para captar características não-lineares
    layers.Dense(16, activation='relu'),
    # Camada de saída com 4 unidades para classificação
    layers.Dense(4, activation='softmax')
])

# Compilação do modelo
model.compile(optimizer=optimizers.Adam(1e-4),
              loss=losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Treinamento do modelo
epochs = 30
historico = model.fit(x_train, y_train,
                      epochs=epochs,
                      validation_data=(x_test, y_test))

import matplotlib.pyplot as plt

# Função para plotar os resultados de acurácia e custo
def plota_resultados(history, epocas, nome):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    intervalo_epocas = range(epocas)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(intervalo_epocas, acc, label='Acurácia do Treino')
    plt.plot(intervalo_epocas, val_acc, label='Acurácia da Validação')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(intervalo_epocas, loss, label='Custo do Treino')
    plt.plot(intervalo_epocas, val_loss, label='Custo da Validação')
    plt.legend(loc='upper right')
    plt.savefig(nome)

plota_resultados(historico, epochs, 'conv_results.png')

"""## LSTM - Redes Recorrentes para Capturar Relações Temporais"""
model = Sequential([
    encoder,
    # Camada de Embedding para representação do texto
    layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=16, mask_zero=False),
    # Camadas LSTM bidirecionais para capturar dependências temporais nos textos
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

# Compilação do modelo LSTM
model.compile(optimizer=optimizers.Adam(1e-4),
              loss=losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Treinamento do modelo LSTM
epochs = 30
historico = model.fit(x_train, y_train,
                      epochs=epochs,
                      validation_data=(x_test, y_test))

plota_resultados(historico, epochs, 'lstm_results.png')

"""## Otimização com Hyperband"""
import keras_tuner as kt

# Função para construir o modelo com hyperparameters
def build_model(hp):
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=hp.Int('embedding_dim', min_value=32, max_value=128, step=32),
            mask_zero=True
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=hp.Int('lstm_units', min_value=32, max_value=128, step=32),
            return_sequences=True
        )),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=hp.Int('lstm_units', min_value=16, max_value=64, step=16)
        )),
        tf.keras.layers.Dense(
            units=hp.Int('dense_units', min_value=32, max_value=128, step=32),
            activation='relu'
        ),
        tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Criação do tuner utilizando Hyperband
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='classification_optimization'
)

"""Definindo a função de validação cruzada com o tuner"""
from sklearn.model_selection import KFold

def run_tuner(x, y, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_index, val_index in kf.split(x):
        x_train_fold, x_val_fold = x[train_index], x[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        # Busca de hiperparâmetros no tuner
        tuner.search(x_train_fold, y_train_fold, epochs=10, validation_data=(x_val_fold, y_val_fold))
    
    # Obtendo os melhores hiperparâmetros
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
          A pesquisa de hiperparâmetros foi concluída. O número ideal de dimensões de incorporação é {best_hps.get('embedding_dim')},
          o número ideal de unidades LSTM é {best_hps.get('lstm_units')}, e
          o número ideal de unidades densas é {best_hps.get('dense_units')},
          e a taxa de abandono ideal é {best_hps.get('dropout')}.
    """)
    return best_hps

best_hps = run_tuner(x_train, y_train, n_splits=5)

"""Treinando a rede otimizada"""
# Construção do modelo final com os melhores hiperparâmetros
final_model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=best_hps.get('embedding_dim'),
        mask_zero=True
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=best_hps.get('lstm_units'),
        return_sequences=True
    )),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=best_hps.get('lstm_units') // 2
    )),
    tf.keras.layers.Dense(
        units=best_hps.get('dense_units'),
        activation='relu'
    ),
    tf.keras.layers.Dropout(rate=best_hps.get('dropout')),
    tf.keras.layers.Dense(4, activation='softmax')
])
final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
final_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

"""Análise de Resultados"""
# Predição e exibição da matriz de confusão
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = final_model.predict(x_test)
y_pred_classes = y_pred.argmax(axis=1)

conf_matrix = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1, 2, 3])
disp.plot(cmap=plt.cm.Blues)
plt.savefig("Confusion_Matrix")
