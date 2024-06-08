import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import logomaker
import pandas as pd
import time

# Create a directory to save the sequences and images if it doesn't exist
output_dir = '../results/sequences'
os.makedirs(output_dir, exist_ok=True)
images_dir = '../results/images'
os.makedirs(images_dir, exist_ok=True)

# Load the trained BPNet model
print("Loading the trained BPNet model...")
model = tf.keras.models.load_model('../results/model/bpnet_model.keras')
print("Model loaded successfully.")

# Define the one-hot encoding function
def one_hot_encode(sequence, max_len):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    one_hot = np.zeros((max_len, 4), dtype=np.int8)
    for i, char in enumerate(sequence[:max_len]):  # Truncate sequence to max_len
        if char in mapping:
            one_hot[i, mapping[char]] = 1
    return one_hot

# Define the function to decode one-hot encoded sequences
def one_hot_decode(one_hot_seq):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    return ''.join([mapping[np.argmax(pos)] for pos in one_hot_seq])

# Function to create sequence logo
def create_sequence_logo(attributions, length):
    counts = np.sum(attributions[:length], axis=0)
    df = pd.DataFrame(counts, columns=['A', 'C', 'G', 'T'])
    return df

# Load validation data
print("Loading validation data...")
X_val = np.load('../data/processed/X.npy', allow_pickle=True)  # Ensure to use the correct file path
print(f"Validation data loaded. Shape: {X_val.shape}")

# Determine the total number of sequences
total_sequences = X_val.shape[0]
print(f"Total number of sequences: {total_sequences}")

# Select a diverse subset of sequences
random.seed(42)  # For reproducibility
num_sequences_to_process = 25
indices = random.sample(range(total_sequences), num_sequences_to_process)
diverse_sequences = [X_val[i] for i in indices]

# Save the selected sequences and their indices to a file
sequences_file = os.path.join(output_dir, 'selected_sequences.txt')
with open(sequences_file, 'w') as f:
    for i, seq in enumerate(diverse_sequences):
        f.write(f">sequence_{indices[i]+1}\n{seq}\n")

# Print the actual sequences (not just one-hot encoded)
print("Sample raw sequences (first 20 nucleotides):")
for i, seq in enumerate(diverse_sequences[:25]):
    print(f"Sequence {indices[i]+1}: {seq[:20]}...")

# Define the maximum sequence length
max_len = 18593

# One-hot encode the selected sequences
print("One-hot encoding validation data...")
X_val_encoded = np.array([one_hot_encode(seq, max_len) for seq in diverse_sequences])
print(f"One-hot encoded validation data. Shape: {X_val_encoded.shape}")

# Save the one-hot encoded sequences to a file
encoded_sequences_file = os.path.join(output_dir, 'one_hot_encoded_sequences.npy')
np.save(encoded_sequences_file, X_val_encoded)

# Verify encoding and decoding
print("Verifying encoding and decoding:")
for i in range(25):
    raw_seq = diverse_sequences[i]
    encoded_seq = X_val_encoded[i]
    decoded_seq = one_hot_decode(encoded_seq[:len(raw_seq)])  # Decode only the original length
    
    print(f"Original Sequence {indices[i]+1}: {raw_seq}")
    print(f"Decoded Sequence {indices[i]+1}:  {decoded_seq}")
    print(f"Match: {raw_seq.upper() == decoded_seq.upper()}\n")  # Normalize case for comparison

# Check model predictions for the first few sequences
print("Model predictions for the first few sequences:")
for i in range(25):
    input_sequence = X_val_encoded[i].astype(np.float32)
    input_sequence = np.expand_dims(input_sequence, axis=0)
    prediction = model.predict(input_sequence)
    print(f"Prediction for sequence {indices[i]+1}: {prediction}")

# Define Integrated Gradients function compatible with TensorFlow 2.x
@tf.function
def compute_gradients(inputs, model):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs)
    grads = tape.gradient(predictions, inputs)
    return grads

def integrated_gradients(inputs, baseline, model, steps=50):
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)]
    avg_grads = np.mean([compute_gradients(tf.convert_to_tensor(x, dtype=tf.float32), model).numpy() for x in scaled_inputs], axis=0)
    integrated_grads = (inputs - baseline) * avg_grads
    return integrated_grads

# Prepare baseline (zero array of same shape as input)
baseline = np.zeros((max_len, 4), dtype=np.float32)

# Compute attributions for the selected sequences and save them
all_attributions = []

start_time = time.time()

for seq_idx in range(num_sequences_to_process):
    input_sequence = X_val_encoded[seq_idx].astype(np.float32)  # Ensure dtype is float32
    input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension
    attributions = integrated_gradients(input_sequence, baseline, model)
    all_attributions.append(attributions[0])
    
    if seq_idx % 1 == 0:  # Print status every sequence
        elapsed_time = time.time() - start_time
        print(f"Processed {seq_idx+1} sequences out of {num_sequences_to_process} in {elapsed_time:.2f} seconds.")

total_time = time.time() - start_time
print(f"Total time taken to compute attributions for {num_sequences_to_process} sequences: {total_time:.2f} seconds")

# Convert attributions list to numpy array and save
attributions_file = os.path.join(output_dir, 'attributions_subset.npy')
all_attributions = np.array(all_attributions)
np.save(attributions_file, all_attributions)
print("Attributions for the subset saved successfully.")

# Verify that the number of attributions matches the number of selected sequences
if len(diverse_sequences) != all_attributions.shape[0]:
    raise ValueError("Mismatch between number of selected sequences and attributions.")
else:
    print("Number of selected sequences matches the attributions.")
