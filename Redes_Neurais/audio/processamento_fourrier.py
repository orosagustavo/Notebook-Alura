import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, callbacks
import dataProcess as dp  # Importa o módulo de processamento

# Configurando GPU (opcional)
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

# Caminho do arquivo e preparação dos dados
gz_path = 'dados/dataset_commands-002.gz'
train_dataset, train_size, val_dataset, val_size = dp.prepare_audio_dataset(gz_path)

# Função para aplicar a transformada de Fourier
def fourrier(formadeonda):
    espectro = tf.signal.stft(formadeonda, frame_length=255, frame_step=128)  # Short-Time Fourier Transform
    espectro = tf.abs(espectro)  # Pega a magnitude do espectro
    espectro = espectro[..., tf.newaxis]  # Adiciona uma dimensão extra para compatibilidade
    return espectro

# Transformando os dados de treino e validação para o formato de espectrograma
def get_spectogram_label_id(audio, label):
    espectro = fourrier(audio)
    return espectro, label

train_spec = train_dataset.map(map_func=get_spectogram_label_id, num_parallel_calls=tf.data.AUTOTUNE)
val_spec = val_dataset.map(map_func=get_spectogram_label_id, num_parallel_calls=tf.data.AUTOTUNE)

# Normalizando os dados
norm_layer = tf.keras.layers.Normalization()
for spectrogram, _ in train_spec.take(1):
    norm_layer.adapt(spectrogram)
    input_shape = spectrogram.shape[1:]  # Armazena o formato da entrada para o modelo

# Implementação de camada de Atenção
# Implementação de camada de Atenção
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)  # Chamada ao construtor da superclasse
        self.ratio = ratio
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()

    def build(self, input_shape):
        # Definindo camadas densa usadas na atenção
        self.fc1 = layers.Dense(units=input_shape[-1] // self.ratio, activation='relu')
        self.fc2 = layers.Dense(units=input_shape[-1], activation='sigmoid')
        super(ChannelAttention, self).build(input_shape)  # Finaliza o build da camada

    def call(self, inputs):
        # Aplicando pooling e atenção aos canais
        avg_out = self.avg_pool(inputs)
        max_out = self.max_pool(inputs)
        avg_out = self.fc2(self.fc1(avg_out))
        max_out = self.fc2(self.fc1(max_out))
        out = avg_out + max_out
        out = tf.expand_dims(tf.expand_dims(out, axis=1), axis=1)
        return inputs * out


# Construindo a rede neural
model_spectrogram = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Resizing(32, 32),
    norm_layer,

    layers.Conv2D(32, 3, activation='relu'),
    ChannelAttention(ratio=8),

    layers.Conv2D(64, 3, activation='relu'),
    ChannelAttention(ratio=8),

    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(36, activation='softmax')
])

# Compilando o modelo
model_spectrogram.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy']
)

# Configurando o checkpoint para interromper o treinamento se a variação de accuracy não passar de 0.01 em 10 epochs
early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=10, restore_best_weights=True)

# Treinando o modelo
history_spectogram = model_spectrogram.fit(train_spec, 
                                          epochs=200, 
                                          validation_data=val_spec, 
                                          callbacks=[early_stopping])

# Salvando o gráfico do histórico de treinamento
dp.plot_history(history_spectogram, 'Atention_cnn')
model_spectrogram.save('models/Atention_Cnn.keras')
