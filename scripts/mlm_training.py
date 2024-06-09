#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load data
print("Loading data...")
X = np.load('../data/processed/X.npy')
y = np.load('../data/processed/y.npy')
print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")

# Use a smaller subset of data for testing
X_subset = X[:1000]
y_subset = y[:1000]
print(f"Using a subset of data: X_subset shape: {X_subset.shape}, y_subset shape: {y_subset.shape}")

# Print a few samples to verify
print("Sample sequences:")
for i in range(3):
    print(f"Sequence {i}: {X_subset[i][:50]}...")  # Print the first 50 nucleotides

print("Sample labels:")
print(y_subset[:10])  # Print the first 10 labels

# Normalize labels
print("Normalizing labels...")
y_subset = (y_subset - np.mean(y_subset)) / np.std(y_subset)
print("Labels normalized.")

# Encode sequences as one-hot (assuming the model expects one-hot encoded input)
def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3, 'N': 4, 'n': 4}
    one_hot = np.zeros((len(seq), 5), dtype=int)
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1
    return one_hot[:, :4]  # Exclude 'N' column for 4-channel output

print("Applying one-hot encoding...")
# Apply one-hot encoding
X_encoded = [one_hot_encode(seq) for seq in X_subset]
print("One-hot encoding applied.")

print("Padding sequences...")
# Pad sequences to the same length
X_padded = pad_sequences(X_encoded, padding='post', dtype='float32')
print(f"Shape of padded data: {X_padded.shape}")

# Split data
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X_padded, y_subset, test_size=0.2, random_state=42)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")

# Create BPNet-like model with adjusted learning rate, increased complexity, and dropout
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

print("Creating BPNet model...")
# Ensure the input length is correctly set
input_length = X_train.shape[1]
model = create_bpnet_model(input_length)
print("BPNet model created.")

# Use a smaller batch size to reduce memory usage
batch_size = 8

# Custom callback to plot the loss after each epoch
class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'val_loss': []}
        
    def on_epoch_end(self, epoch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        
        clear_output(wait=True)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.show()

# Configure EarlyStopping and ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('../results/model/bpnet_model.keras', monitor='val_loss', save_best_only=True)

print("Starting model training...")
# Train model with early stopping and model checkpointing
history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[PlotLosses(), early_stopping, model_checkpoint])
print("Model training completed.")

# Save training history
np.save('../results/model/training_history.npy', history.history)
print("Training history saved.")


# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
print("TensorFlow version:", tf.__version__)


# In[ ]:




