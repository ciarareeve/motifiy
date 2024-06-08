import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse

def train_model(data_dir, output_dir):
    # Load data
    X = np.load(f'{data_dir}/X.npy')
    y = np.load(f'{data_dir}/y.npy')

    # Normalize labels
    y = (y - np.mean(y)) / np.std(y)

    # Encode sequences as one-hot
    def one_hot_encode(seq):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3, 'N': 4, 'n': 4}
        one_hot = np.zeros((len(seq), 5), dtype=int)
        for i, nucleotide in enumerate(seq):
            if nucleotide in mapping:
                one_hot[i, mapping[nucleotide]] = 1
        return one_hot[:, :4]  # Exclude 'N' column for 4-channel output

    # Apply one-hot encoding
    X_encoded = [one_hot_encode(seq) for seq in X]

    # Pad sequences to the same length
    X_padded = pad_sequences(X_encoded, padding='post', dtype='float32')

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_padded, y, test_size=0.2, random_state=42)

    # Create BPNet-like model
    def create_bpnet_model(input_length):
        inputs = layers.Input(shape=(input_length, 4))
        x = layers.Conv1D(128, kernel_size=24, activation='relu', dilation_rate=1)(inputs)
        x = layers.Conv1D(128, kernel_size=24, activation='relu', dilation_rate=2)(x)
        x = layers.Dropout(0.5)(x)  # Add Dropout layer
        x = layers.Conv1D(128, kernel_size=24, activation='relu', dilation_rate=4)(x)
        x = layers.Conv1D(128, kernel_size=24, activation='relu', dilation_rate=8)(x)
        x = layers.GlobalMaxPooling1D()(x)
        outputs = layers.Dense(1)(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mean_squared_error'])
        return model

    # Ensure the input length is correctly set
    input_length = X_train.shape[1]
    model = create_bpnet_model(input_length)

    # Use a smaller batch size to reduce memory usage
    batch_size = 8

    # Custom callback to plot the loss after each epoch
    class PlotLosses(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            clear_output(wait=True)
            plt.plot(self.model.history.history['loss'], label='Training Loss')
            plt.plot(self.model.history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.legend()
            plt.show()

    # Configure EarlyStopping and ModelCheckpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f'{output_dir}/bpnet_model.h5', monitor='val_loss', save_best_only=True)

    # Train model with early stopping and model checkpointing
    history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[PlotLosses(), early_stopping, model_checkpoint])

    # Save training history
    np.save(f'{output_dir}/training_history.npy', history.history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a BPNet-like model on ChIP-seq data.')
    parser.add_argument('--data_dir', required=True, help='Directory containing the processed data')
    parser.add_argument('--output_dir', required=True, help='Directory to save the trained model and history')
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output_dir)

