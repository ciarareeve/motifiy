#!/usr/bin/env python
# coding: utf-8

# In[29]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Load the trained BPNet model
print("Loading the trained BPNet model...")
model = tf.keras.models.load_model('../results/model/bpnet_model.keras')
print("Model loaded successfully.")

# Define the one-hot encoding function
def one_hot_encode(sequence, max_len):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    one_hot = np.zeros((max_len, 4), dtype=np.int8)
    for i, char in enumerate(sequence[:max_len]):
        if char in mapping:
            one_hot[i, mapping[char]] = 1
    return one_hot

# Load validation data
print("Loading validation data...")
X_val = np.load('../data/processed/X.npy', allow_pickle=True)  # Ensure to use the correct file path
print(f"Validation data loaded. Shape: {X_val.shape}")

# Determine the total number of sequences
total_sequences = X_val.shape[0]
print(f"Total number of sequences to be processed: {total_sequences}")

# Define the maximum sequence length
max_len = 18593

# One-hot encode validation data
print("One-hot encoding validation data...")
X_val_encoded = np.array([one_hot_encode(seq, max_len) for seq in X_val])
print(f"One-hot encoded validation data. Shape: {X_val_encoded.shape}")

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

# Compute attributions for all sequences and save them
all_attributions = []

start_time = time.time()

for seq_idx in range(total_sequences):
    input_sequence = X_val_encoded[seq_idx].astype(np.float32)  # Ensure dtype is float32
    input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension
    attributions = integrated_gradients(input_sequence, baseline, model)
    all_attributions.append(attributions[0])
    
    if seq_idx % 100 == 0:  # Print status every 100 sequences
        elapsed_time = time.time() - start_time
        print(f"Processed {seq_idx+1} sequences out of {total_sequences} in {elapsed_time:.2f} seconds.")

total_time = time.time() - start_time
print(f"Total time taken to compute attributions for all sequences: {total_time:.2f} seconds")

# Convert attributions list to numpy array and save
all_attributions = np.array(all_attributions)
np.save('../results/attributions/attributions.npy', all_attributions)
print("Attributions saved successfully.")

# Example visualization for the first n sequences
n = 25  # Specify the number of sequences to visualize
for i in range(n):
    norm_attributions = all_attributions[i] / np.max(np.abs(all_attributions[i]))
    plt.figure(figsize=(10, 6))
    plt.plot(norm_attributions)  # Remove batch dimension for plotting
    plt.title(f'Integrated Gradients Attributions for Sequence {i + 1} (Normalized)')
    plt.xlabel('Position')
    plt.ylabel('Normalized Attribution Score')
    plt.xlim(0, 400)  # Zoom in
    plt.show()


# In[6]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Load the trained BPNet model
print("Loading the trained BPNet model...")
model = tf.keras.models.load_model('../results/model/bpnet_model.keras')
print("Model loaded successfully.")

# Define the one-hot encoding function
def one_hot_encode(sequence, max_len):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    one_hot = np.zeros((max_len, 4), dtype=np.int8)
    for i, char in enumerate(sequence[:max_len]):
        if char in mapping:
            one_hot[i, mapping[char]] = 1
    return one_hot

# Load validation data
print("Loading validation data...")
X_val = np.load('../data/processed/X.npy', allow_pickle=True)  # Ensure to use the correct file path
print(f"Validation data loaded. Shape: {X_val.shape}")

# Define the maximum sequence length
max_len = 18593

# One-hot encode validation data
print("One-hot encoding validation data...")
X_val_encoded = np.array([one_hot_encode(seq, max_len) for seq in X_val])
print(f"One-hot encoded validation data. Shape: {X_val_encoded.shape}")

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

#roughly 15 seconds to process per sequence.  will take about 10 days
num_sequences_to_process = 100 

# Compute attributions for the specified number of sequences and save them
all_attributions = []

start_time = time.time()

for seq_idx in range(num_sequences_to_process):
    input_sequence = X_val_encoded[seq_idx].astype(np.float32)  # Ensure dtype is float32
    input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension
    seq_start_time = time.time()
    attributions = integrated_gradients(input_sequence, baseline, model)
    all_attributions.append(attributions[0])
    
    if seq_idx % 1 == 0:  # Print status every 1 sequence
        elapsed_time = time.time() - seq_start_time
        print(f"Processed sequence {seq_idx+1} in {elapsed_time:.2f} seconds.")
        
total_time = time.time() - start_time
print(f"Total time taken to compute attributions for {num_sequences_to_process} sequences: {total_time:.2f} seconds")

# Convert attributions list to numpy array and save
all_attributions = np.array(all_attributions)
np.save('../results/attributions/attributions_subset.npy', all_attributions)
print("Attributions for the subset saved successfully.")

# Example visualization for the first n sequences
n = 5  # Specify the number of sequences to visualize
for i in range(n):
    norm_attributions = all_attributions[i] / np.max(np.abs(all_attributions[i]))
    plt.figure(figsize=(10, 6))
    plt.plot(norm_attributions)  # Remove batch dimension for plotting
    plt.title(f'Integrated Gradients Attributions for Sequence {i + 1} (Normalized)')
    plt.xlabel('Position')
    plt.ylabel('Normalized Attribution Score')
    plt.xlim(0, 400)  # Zoom in
    plt.show()


# In[21]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logomaker
import pandas as pd

# Create a function to generate a sequence logo from attributions
def create_sequence_logo(attributions, seq_len):
    # Generate dataframe for logomaker
    logo_df = pd.DataFrame(attributions[:seq_len], columns=['A', 'C', 'G', 'T'])
    return logo_df

# Normalize attributions to span the entire length of the plot
def normalize_attributions(attributions):
    max_val = np.max(np.abs(attributions))
    if max_val != 0:
        return attributions / max_val
    else:
        return attributions

# Example visualization for the first n sequences using logomaker
n = 5  # Specify the number of sequences to visualize
plot_length = 400  # Adjust the plot length to ensure motifs span the entire plot
for i in range(n):
    normalized_attributions = normalize_attributions(all_attributions[i])
    logo_df = create_sequence_logo(normalized_attributions, plot_length)
    plt.figure(figsize=(20, 4))  # Increase the width for better visibility
    logomaker.Logo(logo_df, color_scheme='classic')
    plt.title(f'Integrated Gradients Attributions for Sequence {i + 1}', fontsize=20)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Attribution Score', fontsize=12)
    plt.xticks(ticks=np.arange(0, plot_length, 100), fontsize=8)  # Adjust tick spacing and size
    plt.yticks(fontsize=12)
    plt.savefig(f'../results/attributions/logo_sequence_{i+1}.png', bbox_inches='tight')  # Save the figure
    plt.show()


# In[38]:


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
        f.write(f">sequence_{indices[i]+1}
{seq}
")

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
    print(f"Match: {raw_seq.upper() == decoded_seq.upper()}
")  # Normalize case for comparison

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


# In[46]:


import numpy as np
import matplotlib.pyplot as plt
import logomaker
import pandas as pd
import os

# Create directories to save images if they don't exist
images_dir = '../results/images'
os.makedirs(images_dir, exist_ok=True)

# Load attributions
print("Loading attributions...")
attributions = np.load('../results/sequences/attributions_subset.npy')  # Ensure the correct file path
print(f"Attributions shape: {attributions.shape}")

# Create function to generate sequence logo DataFrame
def create_sequence_logo(attributions, length):
    logo_df = pd.DataFrame(attributions[:length], columns=['A', 'C', 'G', 'T'])
    return logo_df

# Plot normalized attributions and sequence logos for each sequence

og_seq = [41906, 7297, 1640, 48599, 18025, 16050, 14629, 9145,48266, 6718,44349, 48541, 35742, 5698, 38699, 27652, 2083, 1953, 6141, 14329, 15248, 33119, 39454, 1740, 36782]

# Iterate over each sequence and create the plot
for i in range(og_seq):
    norm_attributions = attributions[i] / np.max(np.abs(attributions[i]))
    
    # Plot normalized attributions
    plt.figure(figsize=(10, 6))
    plt.plot(norm_attributions[:400])  # Plot only the first 400 positions for better visibility
    plt.title(f'Integrated Gradients Attributions for Original Sequence {og_seq[i]} (Normalized)')
    plt.xlabel('Position')
    plt.ylabel('Normalized Attribution Score')
    plt.savefig(os.path.join(images_dir, f'attributions_sequence_{og_seq[i]}.png'))
    plt.show()
    
    # # Create and plot sequence logo
    # logo_df = create_sequence_logo(norm_attributions, 400)  # Adjust the plot length as needed
    # plt.figure(figsize=(20, 4))  # Increase the width for better visibility
    # logomaker.Logo(logo_df, color_scheme='classic')
    # plt.title(f'Sequence Logo for Sequence {i + 1}')
    # plt.ylim(-0.5, 0.5)  # Adjust y-axis limits for better visibility
    # plt.xlim(50, 100)  # Adjust x-axis limits to match the plot length
    # plt.savefig(os.path.join(images_dir, f'sequence_logo_sequence_{i + 1}.png'))
    # plt.show()

print("Plots saved.")


# In[48]:


og_seq = [41906, 7297, 1640, 48599, 18025, 16050, 14629, 9145,48266, 6718,44349, 48541, 35742, 5698, 38699, 27652, 2083, 1953, 6141, 14329, 15248, 33119, 39454, 1740, 36782]

n = attributions.shape[0]

for i in range(n):
    norm_attributions = attributions[i] / np.max(np.abs(attributions[i]))
    
    # Plot normalized attributions
    plt.figure(figsize=(10, 6))
    plt.plot(norm_attributions[:400])  # Plot only the first 400 positions for better visibility
    plt.title(f'Integrated Gradients Attributions for Original Sequence {og_seq[i]} (Normalized)')
    plt.xlabel('Position')
    plt.ylabel('Normalized Attribution Score')
    plt.savefig(os.path.join(images_dir, f'attributions_sequence_{og_seq[i]}.png'))
    plt.show()


# In[ ]:




