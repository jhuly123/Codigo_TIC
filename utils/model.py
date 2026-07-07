# model.py — arquitectura LSTM y callbacks de entrenamiento, centralizados
# para que todos los experimentos y el entrenamiento en producción usen
# exactamente la misma base.

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                      MultiHeadAttention, GlobalAveragePooling1D,
                                      LayerNormalization, Input, Add)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from .config import N_FRAMES, N_KEYPOINTS


def construir_lstm(n_clases: int,
                   lr: float = 0.0005,
                   dropout: float = 0.4,
                   regularizacion: float = 0.0,
                   capa_densa: bool = True,
                   unidades_1: int = 64,
                   unidades_2: int = 32) -> tf.keras.Model:
    """
    Clasificador LSTM de dos capas para secuencias de keypoints (60, 225).

      LSTM(unidades_1) → BN → Dropout → LSTM(unidades_2) → BN → Dropout
      [→ Dense(128, relu) → Dropout]   ← solo si capa_densa=True
      → Dense(n_clases, softmax)
    """
    reg   = l2(regularizacion) if regularizacion > 0 else None
    capas = [
        LSTM(unidades_1, return_sequences=True, input_shape=(N_FRAMES, N_KEYPOINTS),
             kernel_regularizer=reg),
        BatchNormalization(), Dropout(dropout),
        LSTM(unidades_2, kernel_regularizer=reg),
        BatchNormalization(), Dropout(dropout),
    ]
    if capa_densa:
        capas += [Dense(128, activation='relu'), Dropout(dropout + 0.1)]
    capas.append(Dense(n_clases, activation='softmax'))

    model = Sequential(capas)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def construir_lstm_atencional(n_clases: int,
                               lr: float = 0.0005,
                               dropout: float = 0.4,
                               num_heads: int = 4) -> tf.keras.Model:
    """
    Clasificador LSTM + Atención Multi-Cabeza para secuencias de keypoints (60, 225).
    Comparable a Jiang et al. (2021), "Sign Language Recognition Based on
    Feature Extraction", IEEE Access.

      LSTM(64, return_sequences) → BN → Dropout
      MultiHeadAttention(num_heads) + residual → LayerNorm
      GlobalAveragePooling1D → Dense(64, relu) → Dropout → Dense(n_clases, softmax)

    El bloque de atención permite ponderar dinámicamente qué frames son más
    discriminativos, en vez de depender solo del estado final del LSTM.
    num_heads debe dividir exactamente 64.
    """
    inp = Input(shape=(N_FRAMES, N_KEYPOINTS))

    x = LSTM(64, return_sequences=True)(inp)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=64 // num_heads)(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    out = Dense(n_clases, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def callbacks_entrenamiento(checkpoint_path: str = None,
                             patience_stop: int = 20,
                             patience_lr: int = 8) -> list:
    """
    Callbacks estándar de entrenamiento:
      - EarlyStopping (restaura mejores pesos al detener)
      - ReduceLROnPlateau (divide lr por 2 tras patience_lr épocas sin mejora)
      - ModelCheckpoint (solo si se proporciona checkpoint_path)
    """
    cbs = [
        EarlyStopping(monitor='val_loss', patience=patience_stop,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=patience_lr, min_lr=1e-5, verbose=1),
    ]
    if checkpoint_path:
        cbs.append(ModelCheckpoint(checkpoint_path, monitor='val_accuracy',
                                   save_best_only=True, verbose=0))
    return cbs
