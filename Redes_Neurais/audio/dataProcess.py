import pandas as pd
import tensorflow as tf
import pathlib
import gzip
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Funções para processamento de dados tabulares
def load_data(filepath):
    """
    Carrega os dados a partir de um arquivo CSV.
    
    Args:
        filepath (str): Caminho para o arquivo CSV.
    
    Returns:
        pd.DataFrame: DataFrame contendo os dados carregados.
    """
    return pd.read_csv(filepath)

def clean_data(dataframe):
    """
    Limpa os dados removendo valores nulos.
    
    Args:
        dataframe (pd.DataFrame): DataFrame com os dados brutos.
    
    Returns:
        pd.DataFrame: DataFrame sem valores nulos.
    """
    return dataframe.dropna()

def normalize_data(features):
    """
    Normaliza os dados usando StandardScaler.
    
    Args:
        features (pd.DataFrame): DataFrame com as features a serem normalizadas.
    
    Returns:
        np.ndarray: Array com as features normalizadas.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def split_data(features, target, test_size=0.2, random_state=42):
    """
    Divide os dados em treino e teste.
    
    Args:
        features (np.ndarray): Features normalizadas.
        target (pd.Series): Target correspondente.
        test_size (float): Proporção dos dados para o conjunto de teste.
        random_state (int): Semente para reprodução dos resultados.
    
    Returns:
        tuple: Tupla contendo X_train, X_test, y_train, y_test.
    """
    return train_test_split(features, target, test_size=test_size, random_state=random_state)

# Funções para processamento de áudio
def le_arquivos(gz_path):
    """
    Lê e extrai arquivos comprimidos .gz para um diretório temporário.
    
    Args:
        gz_path (str): Caminho para o arquivo .gz.
    
    Returns:
        tuple: Lista de caminhos de arquivos de áudio e suas respectivas labels.
    """
    extracted_path = '/tmp/dataset_commands'
    with gzip.open(gz_path, 'rb') as f_in:
        with open(extracted_path + '.tar', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    # Extrai o arquivo .tar resultante
    shutil.unpack_archive(extracted_path + '.tar', extracted_path)
    data_dir = pathlib.Path(extracted_path)
    # Lista todos os arquivos de áudio e suas respectivas labels
    all_audio_paths = list(data_dir.glob('*/**/*.wav'))
    all_labels = [path.parent.name for path in all_audio_paths]
    # Converte os caminhos para strings
    all_audio_paths = [str(path) for path in all_audio_paths]
    return all_audio_paths, all_labels

def load_and_process_audio(filename, max_length=16000):
    """
    Carrega o arquivo de áudio e aplica resampling e padding.
    
    Args:
        filename (str): Caminho para o arquivo de áudio.
        max_length (int): Tamanho máximo do áudio em amostras.
    
    Returns:
        tf.Tensor: Tensor contendo o áudio processado.
    """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)  # Remove dimensão extra
    
    # Função para realizar resampling usando SciPy
    def scipy_resample(wav, sample_rate):
        if sample_rate != 16000:
            wav = resample(wav, int(16000 / sample_rate * len(wav)))
        return wav

    # Usa tf.py_function para envolver o resampling em TensorFlow
    wav = tf.py_function(scipy_resample, [wav, sample_rate], tf.float32)
    audio_length = tf.shape(wav)[0]
    # Aplica padding ou corta o sinal para ter um tamanho fixo
    if audio_length > max_length:
        wav = wav[:max_length]
    else:
        pad_length = max_length - audio_length
        paddings = [[0, pad_length]]
        wav = tf.pad(wav, paddings, "CONSTANT")
    return tf.reshape(wav, [max_length])

def process_path(file_path, label):
    """
    Carrega o áudio e o associa à sua label.
    
    Args:
        file_path (str): Caminho para o arquivo de áudio.
        label (int): Label associada ao áudio.
    
    Returns:
        tuple: Áudio processado e sua label.
    """
    audio = load_and_process_audio(file_path)
    return audio, label

def paths_and_labels_to_dataset(audio_paths, labels):
    """
    Cria um dataset a partir dos caminhos dos arquivos de áudio e suas labels.
    
    Args:
        audio_paths (list): Lista de caminhos dos arquivos de áudio.
        labels (list): Lista de labels associadas aos arquivos de áudio.
    
    Returns:
        tf.data.Dataset: Dataset contendo os áudios e suas labels.
    """
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    audio_label_ds = tf.data.Dataset.zip((path_ds, label_ds))
    # Aplica o processamento dos caminhos de arquivos
    return audio_label_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

def prepare_for_training(ds, batch_size=32, shuffle_buffer_size=1000):
    """
    Prepara o dataset para o treinamento, aplicando shuffle, batching e prefetching.
    
    Args:
        ds (tf.data.Dataset): Dataset a ser preparado.
        batch_size (int): Tamanho dos lotes.
        shuffle_buffer_size (int): Tamanho do buffer de shuffle.
    
    Returns:
        tf.data.Dataset: Dataset preparado para treinamento.
    """
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)  # Embaralha os dados
    ds = ds.batch(batch_size)  # Agrupa os dados em lotes
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)  # Faz prefetch para melhorar a eficiência
    return ds

def encode_labels(labels):
    """
    Codifica as labels como inteiros.
    
    Args:
        labels (list): Lista de labels em formato de string.
    
    Returns:
        np.ndarray: Labels codificadas como inteiros.
    """
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(labels)

def prepare_audio_dataset(gz_path, batch_size=32, shuffle_buffer_size=1000, validation_split=0.2):
    """
    Junta todos os passos de processamento de dados de áudio e retorna os datasets de treino e validação.
    
    Args:
        gz_path (str): Caminho para o arquivo .gz contendo os dados de áudio.
        batch_size (int): Tamanho dos lotes para o treinamento.
        shuffle_buffer_size (int): Tamanho do buffer de shuffle.
        validation_split (float): Proporção dos dados a serem usados para validação.
    
    Returns:
        tuple: Datasets de treino e validação preparados para treinamento.
    """
    # Carrega e extrai os arquivos de áudio
    audio_paths, labels = le_arquivos(gz_path)
    
    # Codifica as labels como inteiros
    labels_encoded = encode_labels(labels)
    
    # Cria o dataset a partir dos caminhos dos arquivos de áudio e labels
    complete_dataset = paths_and_labels_to_dataset(audio_paths, labels_encoded)
    
    # Divide o dataset em treino e validação
    total_size = len(audio_paths)
    val_size = int(validation_split * total_size)
    train_size = total_size - val_size
    
    complete_dataset = complete_dataset.shuffle(buffer_size=total_size, seed=42)
    train_dataset = complete_dataset.take(train_size)
    val_dataset = complete_dataset.skip(train_size)
    
    # Prepara os datasets para treinamento
    train_dataset = prepare_for_training(train_dataset, batch_size=batch_size, shuffle_buffer_size=shuffle_buffer_size)
    val_dataset = prepare_for_training(val_dataset, batch_size=batch_size, shuffle_buffer_size=shuffle_buffer_size)
    
    return train_dataset, train_size, val_dataset, val_size 

# Função para plotar o histórico de treino
def plot_history(history, name='nome'):
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
    plt.savefig("graficos/"+name)
