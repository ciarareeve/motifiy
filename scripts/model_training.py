# scripts/model_training.py

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow as tf

def create_bpnet_model(input_length):
    model = models.Sequential()
    model.add(layers.Conv1D(64, kernel_size=24, activation='relu', input_shape=(input_length, 4)))
    model.add(layers.Conv1D(64, kernel_size=24, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(128, kernel_size=24, activation='relu'))
    model.add(layers.Conv1D(128, kernel_size=24, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    X = np.load('data/processed/X.npy')
    y = np.load('data/processed/y.npy')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    input_length = X_train.shape[1]

    model = create_bpnet_model(input_length)
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    model.save('results/model/bpnet_model.h5')
    np.save('results/model/training_history.npy', history.history)

if __name__ == "__main__":
    main()

