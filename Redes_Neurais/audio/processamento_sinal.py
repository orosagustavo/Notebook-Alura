# %% [markdown]
# # Aula 1 - Introduzindo o reconhecimento de áudio

# %% [markdown]
# ## Vídeo 1.2 - Carregando os dados

# %%
gz_path = 'dados/dataset_commands-002.gz'

# %%
#!pip install tensorflow

# %%
import tensorflow as tf
import pathlib
import os
import gzip
import shutil
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=7250)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f'{len(gpus)} GPU física disponível, {len(logical_gpus)} GPU lógica configurada.')
    except RuntimeError as e:
        # A configuração de dispositivos deve ser feita antes de qualquer operação do TensorFlow.
        print(e)

# %%
def le_arquivos(gz_path):
    # Diretório onde os dados serão extraídos
    extracted_path = '/tmp/dataset_commands'
    # Extrair o arquivo .gz
    with gzip.open(gz_path, 'rb') as f_in:
        with open(extracted_path + '.tar', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Extrair o arquivo .tar resultante
    shutil.unpack_archive(extracted_path + '.tar', extracted_path)

    # Caminho para os dados extraídos
    data_dir = pathlib.Path(extracted_path)

    # Listar todos os arquivos de áudio e suas labels
    all_audio_paths = list(data_dir.glob('*/**/*.wav'))
    all_labels = [path.parent.name for path in all_audio_paths]

    # Converter caminhos para strings
    all_audio_paths = [str(path) for path in all_audio_paths]

    return all_audio_paths, all_labels

# %%
all_audio_paths, all_labels = le_arquivos(gz_path)

# %%
np.unique(all_labels)

# %%
np.unique(all_labels).shape

# %%
import matplotlib.pyplot as plt

# %%
example_audio_path = all_audio_paths[0]

# %%
# Carregar o arquivo de áudio
audio_binary = tf.io.read_file(example_audio_path)
audio, _ = tf.audio.decode_wav(audio_binary)
audio = tf.squeeze(audio, axis=-1)

# %%
# Plotar a forma de onda
plt.figure(figsize=(10, 6))
plt.plot(audio.numpy())
plt.title(f'Forma de onda para {example_audio_path}')
plt.xlabel('Amostras')
plt.ylabel('Amplitude')
plt.show()

# %% [markdown]
# ## Vídeo 1.3 - Processando os dados de áudio

# %%
from scipy.signal import resample


# %%
# Função para carregar e processar o áudio com resampling usando SciPy
def load_and_process_audio(filename, max_length=16000):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    
    # Função de resampling usando SciPy
    def scipy_resample(wav, sample_rate):
        if sample_rate != 16000:
            wav = resample(wav, int(16000 / sample_rate * len(wav)))
        return wav

    # Usar tf.py_function para envolver a operação de resampling
    wav = tf.py_function(scipy_resample, [wav, sample_rate], tf.float32)
    
    # Adicionar padding ou cortar os sinais de áudio
    audio_length = tf.shape(wav)[0]
    if audio_length > max_length:
        wav = wav[:max_length]
    else:
        pad_length = max_length - audio_length
        paddings = [[0, pad_length]]
        wav = tf.pad(wav, paddings, "CONSTANT")
    
    return tf.reshape(wav, [max_length])

# %%
# Função para processar o caminho do arquivo de áudio e sua label
def process_path(file_path, label):
    audio = load_and_process_audio(file_path)
    return audio, label

# %%
# Criar um dataset do TensorFlow
def paths_and_labels_to_dataset(audio_paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    audio_label_ds = tf.data.Dataset.zip((path_ds, label_ds))
    return audio_label_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

# %%
# Função para preparar o dataset para o treinamento
def prepare_for_training(ds, batch_size=32, shuffle_buffer_size=1000):
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

# %%
from sklearn.preprocessing import LabelEncoder
# Codificar as labels como inteiros
label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)

# %%
# Conjunto completo de dados
complete_dataset = paths_and_labels_to_dataset(all_audio_paths, all_labels_encoded)

# %% [markdown]
# ## Vídeo 1.4 - Treinando a rede

# %%
# Dividir o dataset em treino e validação
total_size = len(all_audio_paths)
val_size = int(0.2 * total_size)
train_size = total_size - val_size

# %%
complete_dataset = complete_dataset.shuffle(buffer_size=total_size, seed=42)
train_dataset = complete_dataset.take(train_size)
val_dataset = complete_dataset.skip(train_size)

# %%
train_dataset = prepare_for_training(train_dataset)
val_dataset = prepare_for_training(val_dataset)

# %%
from tensorflow.keras import layers, models, callbacks

# %%
# Treinamento no domínio do tempo
model_time_domain = models.Sequential([
    layers.Input(shape=(16000, 1)),
    layers.Conv1D(16, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(32, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(36, activation='softmax')
])

# %%
model_time_domain.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

# %%
early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=10, restore_best_weights=True)
history_time_domain = model_time_domain.fit(train_dataset, epochs=250, validation_data=val_dataset,
                                            callbacks = [early_stopping])


# %%
def plot_history(history):
    # Resumo do histórico de precisão
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Acurácia do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend(loc='lower right')

    # Resumo do histórico de perda
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Perda do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("results_cnn")


# %%
plot_history(history_time_domain)

model_time_domain.save("signal_cnn.keras")

