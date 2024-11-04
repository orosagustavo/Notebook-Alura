import pandas as pd
import tensorflow as tf

url = 'https://github.com/allanspadini/curso-tensorflow-proxima-palavra/raw/main/dados/train.zip'

df = pd.read_csv(url, header=None, names=['ClassIndex', 'Título', 'Descrição'])

df['Texto'] = df['Título'] + '' + df['Descrição']

import random

random.seed(42)
df_sample = df.sample(n=1000)

"""
Criando um `corpus` que é o nosso conjunto completo de textos, convertendo o conteúdo da coluna de texto para uma lista.
"""
corpus = df_sample['Texto'].tolist()

from tensorflow.keras.layers import TextVectorization
max_vocab_size = 20000
max_sequence_length = 50

vectorizer = TextVectorization(max_tokens=max_vocab_size, output_sequence_length=max_sequence_length, output_mode='int')
vectorizer.adapt(corpus)

tokenized_corpus = vectorizer(corpus)

import pickle

with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

"""
Iremos passar sequências de texto para a rede neural ser treinada
"""
input_sequences = []

for token_list in tokenized_corpus.numpy():
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Colocando zeros para regularizar a entrada da rede
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def prepare_sequences(sequences):
    """
    Prepara as sequências para o modelo, removendo zeros à direita, adicionando padding à esquerda, truncado sequências longas e removendo sequências repetidas.

    Args:
        sequences: Um array de sequências (listas ou arrays NumPy).

    Returns:
        Um array NumPy 2D com as sequências preparadas.
    """

    # Remover zeros à direita de cada sequência
    sequences_without_trailing_zeros = []
    for seq in sequences:
        last_nonzero_index = np.argmax(seq[::-1] != 0)
        if last_nonzero_index == 0 and seq[-1] == 0:
            sequences_without_trailing_zeros.append(np.array([0]))
        else:
            sequences_without_trailing_zeros.append(seq[:-last_nonzero_index or None])

    # Remover sequências repetidas
    unique_sequences = []
    for seq in sequences_without_trailing_zeros:
        if seq.tolist() not in unique_sequences:  # Verifica se a sequência já está na lista
            unique_sequences.append(seq.tolist())  # Adiciona à lista se for única

    # Encontrar o comprimento máximo das sequências sem zeros à direita
    max_sequence_len = max(len(seq) for seq in unique_sequences)

    # Adicionar padding à esquerda para garantir o mesmo comprimento
    padded_sequences = pad_sequences(unique_sequences, maxlen=max_sequence_len, padding='pre', truncating='post')

    return padded_sequences

input_sequences_prepared = prepare_sequences(input_sequences)
print(input_sequences_prepared)

"""
Criando e treinando o modelo
"""

# Dividindo a sequência como se fosse um problema de classificação multiclasse
x = input_sequences_prepared[:,:-1]
y = input_sequences_prepared[:,-1]

y = tf.keras.utils.to_categorical(y, num_classes = max_vocab_size)

# Definir o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=max_vocab_size,
        output_dim=128,
        mask_zero=False
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),  # Camada BiLSTM
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(max_vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

history = model.fit(x, y, epochs=100, verbose=1, batch_size=64)

import matplotlib.pyplot as plt


# Plotar a acurácia e a perda durante o treinamento
plt.plot(history.history['accuracy'])
plt.title('Modelo de precisão')
plt.ylabel('Precisão')
plt.xlabel('Época')
plt.legend(['Acurácia'], loc='upper left')
plt.savefig("Corretor")

"""
Vamos criar a previsão das próximas palavras
"""
def predict_next_words(model, vectorizer, text, max_sequence_len, top_k=3):
    # Vetorizar o texto de entrada
    tokenized_text = vectorizer([text])

    # Remover a dimensão extra adicionada pela vetorização
    tokenized_text = np.squeeze(tokenized_text)

    # Adicionar padding à esquerda
    padded_text = pad_sequences([tokenized_text], maxlen=max_sequence_len, padding='pre')
        
    # Fazer a previsão
    predicted_probs = model.predict(padded_text, verbose=0)[0]  # Remove a dimensão extra adicionada pela previsão
        
    # Obter os índices dos top_k tokens com as maiores probabilidades
    top_k_indices = np.argsort(predicted_probs)[-top_k:][::-1]
        
    # Converter os tokens previstos de volta para palavras
    predicted_words = [vectorizer.get_vocabulary()[index] for index in top_k_indices]
        
    return predicted_words

text = "The FBI is warning consumers against using public phone charging stations in order to"
print("Texto: ", text)
print("Predict: ")
predict_next_words(model, vectorizer, text, 50, top_k=3)


text = "Apple needs to make the iPhone"
print("Texto: ", text)
print("Predict: ")
predict_next_words(model, vectorizer, text, 50, top_k=3)

model.save('corretor.keras')