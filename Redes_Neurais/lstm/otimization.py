url ='https://github.com/allanspadini/curso-tensorflow-proxima-palavra/raw/main/dados/train.zip'

import pandas as pd

dados = pd.read_csv(url, header=None,names=['ClassIndex', 'Título','Descrição'])
dados.head()

dados['ClassIndex'].unique()

dados['ClassIndex'] = dados['ClassIndex'] - 1

dados['Texto'] = dados['Título'] + ' ' + dados['Descrição']
dados.head()

"""Separando os dados de treino e teste"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dados['Texto'].values,
                                                    dados['ClassIndex'].values,
                                                    test_size=0.2, random_state=4256)

"""O próximo passo é transformar o texto em um elemento de um espaço vetorial. Para isso vamos usar a técninca de tokenização"""

import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, losses

# Definindo o máximo de palaras
VOCAB_SIZE = 1000

# Definindo o encoder
encoder = layers.TextVectorization(max_tokens = VOCAB_SIZE)
encoder.adapt(x_train)


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

"""
Passamos os parâmetros para o tuner
"""
import keras_tuner as kt
tuner = kt.Hyperband(
    build_model,
    objective = 'val_accuracy',
    max_epochs = 10,
    factor = 3,
    directory = 'my_dir',
    project_name = 'classification_optimization'
)

"""
Agora podemos definir uma forma de rodar o processo de otimização
"""

from sklearn.model_selection import KFold

def run_tuner(x,y, n_splits):
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)

    for train_index, val_index in kf.split(x):
        x_train_fold, x_val_fold = x[train_index], x[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        tuner.search(x_train_fold, y_train_fold, epochs=10, validation_data = (x_val_fold, y_val_fold))
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
          A pesquisa de hiperparâmetros foi concluída. O número ideal de dimensões de incorporação é {best_hps.get('embedding_dim')},
          o número ideal de unidades LSTM é {best_hps.get('lstm_units')}, e
          o número ideal de unidades densas é {best_hps.get('dense_units')},
          e a taxa de abandono ideal é {best_hps.get('dropout')}.
    """)
    return best_hps

best_hps = run_tuner(x_train, y_train, n_splits=10)