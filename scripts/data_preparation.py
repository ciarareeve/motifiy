#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from pyfaidx import Fasta

# Load genome
genome = Fasta('../data/raw/hg38.fa')
print("Reference genome loaded.")

# Load ChIP-seq data
chip_seq_data = pd.read_csv('../data/raw/chseq2.bed', sep='\t', header=None)
chip_seq_data.columns = ['chrom', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak']
print("ChIP-seq data loaded. Total entries:", len(chip_seq_data))

# Filter out non-human chromosomes
valid_chromosomes = set(genome.keys())
initial_count = len(chip_seq_data)
chip_seq_data = chip_seq_data[chip_seq_data['chrom'].isin(valid_chromosomes)]
filtered_count = len(chip_seq_data)
print(f"Filtered out {initial_count - filtered_count} entries with non-human chromosomes.")
print("Remaining entries after filtering:", filtered_count)

# Extract sequences
sequences = []
for index, row in chip_seq_data.iterrows():
    try:
        sequence = genome[row['chrom']][row['start']:row['end']].seq
        sequences.append(sequence)
    except KeyError as e:
        print(f"Error extracting sequence for row {index}: {e}")

sequences = np.array(sequences)
print("Sequences extracted. Total sequences:", len(sequences))

# Save processed data
np.save('../data/processed/X.npy', sequences)
np.save('../data/processed/y.npy', chip_seq_data['signalValue'].values)
print("Processed data saved.")


# 
# 1. **`data_preparation.ipynb`**
# 
#    **Description**:
#    - This notebook prepares the data by loading the reference genome and ChIP-seq peak data, extracting sequences based on the peak coordinates, and saving the processed data as NumPy arrays.
# 
#    **Input**:
#    - Reference genome file (`hg19.fa`)
#    - ChIP-seq peak file (`chip_seq_data.bed`)
# 
#    **Output**:
#    - Processed sequences (`X.npy`)
#    - Signal values (`y.npy`)
# 
#    **Checks**:
#    - Ensure that `X.npy` and `y.npy` are created in the `data/processed/` directory.
#    - Verify the shape and content of the arrays to ensure they match expectations.
#      ```python
#      # Load and check the processed data
#      X = np.load('../data/processed/X.npy')
#      y = np.load('../data/processed/y.npy')
#      
#      # Check shapes
#      print(X.shape)  # Expected shape: (number of peaks, sequence length)
#      print(y.shape)  # Expected shape: (number of peaks,)
# 
#      # Check content
#      print(X[0])  # Print the first sequence
#      print(y[0])  # Print the first signal value
#      ```
# 
# 2. **`model_training.ipynb`**
# 
#    **Description**:
#    - This notebook trains the BPNet model using the processed data. It splits the data into training and validation sets, creates the model, trains it, and saves the trained model and training history.
# 
#    **Input**:
#    - Processed sequences (`X.npy`)
#    - Signal values (`y.npy`)
# 
#    **Output**:
#    - Trained model (`bpnet_model.h5`)
#    - Training history (`training_history.npy`)
# 
#    **Checks**:
#    - Ensure that `bpnet_model.h5` and `training_history.npy` are created in the `results/model/` directory.
#    - Verify the training history to check for convergence and potential overfitting.
#      ```python
#      # Load and check the training history
#      history = np.load('../results/model/training_history.npy', allow_pickle=True).item()
#      
#      # Plot training and validation loss
#      import matplotlib.pyplot as plt
#      plt.plot(history['loss'], label='Training Loss')
#      plt.plot(history['val_loss'], label='Validation Loss')
#      plt.legend()
#      plt.show()
#      ```
# 
# 3. **`deepLIFT_attribution.ipynb`**
# 
#    **Description**:
#    - This notebook uses the trained BPNet model to calculate feature attributions using DeepLIFT. It saves the attributions as a NumPy array.
# 
#    **Input**:
#    - Trained model (`bpnet_model.h5`)
#    - Validation data (`X_val.npy`)
# 
#    **Output**:
#    - Attributions (`attributions.npy`)
# 
#    **Checks**:
#    - Ensure that `attributions.npy` is created in the `results/attributions/` directory.
#    - Verify the content and shape of the attributions to ensure they are correctly calculated.
#      ```python
#      # Load and check the attributions
#      attributions = np.load('../results/attributions/attributions.npy')
#      
#      # Check shape
#      print(attributions.shape)  # Expected shape: (number of validation samples, sequence length)
# 
#      # Check content
#      print(attributions[0])  # Print the first attribution
#      ```
# 
# 4. **`clustering.ipynb`**
# 
#    **Description**:
#    - This notebook extracts and clusters seqlets based on the calculated attributions to identify motifs. It saves the clustering results.
# 
#    **Input**:
#    - Attributions (`attributions.npy`)
# 
#    **Output**:
#    - Positive clusters (`positive_clusters.npy`)
#    - Negative clusters (`negative_clusters.npy`)
# 
#    **Checks**:
#    - Ensure that `positive_clusters.npy` and `negative_clusters.npy` are created in the `results/clusters/` directory.
#    - Verify the clusters to ensure they make sense.
#      ```python
#      # Load and check the clusters
#      positive_clusters = np.load('../results/clusters/positive_clusters.npy', allow_pickle=True).item()
#      negative_clusters = np.load('../results/clusters/negative_clusters.npy', allow_pickle=True).item()
#      
#      # Check content
#      print(positive_clusters.keys())  # Print the keys of positive clusters
#      print(negative_clusters.keys())  # Print the keys of negative clusters
#      ```
# 
# ### Quality Check for the Model
# 
# To determine if the trained BPNet model is suitable for motif finding:
# 
# 1. **Training and Validation Loss**:
#    - Plot the training and validation loss to check for convergence.
#    - Look for signs of overfitting (e.g., training loss decreases but validation loss increases).
# 
# 2. **Accuracy**:
#    - Check the training and validation accuracy (if applicable) to ensure the model is learning the correct patterns.
# 
# 3. **Attributions**:
#    - Inspect the attributions to ensure they highlight meaningful parts of the sequences.
#    - High attributions should correspond to known motifs or significant regions in the sequences.
# 
# 4. **Cluster Analysis**:
#    - Inspect the clusters to ensure they group similar seqlets together.
#    - Visualize some of the motifs to see if they make biological sense.
# 
# By performing these checks and analyses, you can ensure that each step of the workflow is correctly executed and that the final motifs identified by the model are meaningful and accurate.

# In[2]:


# Load and check the processed data
X = np.load('../data/processed/X.npy')
y = np.load('../data/processed/y.npy')

# Check shapes
print(X.shape)  # Expected shape: (number of peaks, sequence length)
print(y.shape)  # Expected shape: (number of peaks,)

# Check content
print(X[0])  # Print the first sequence
print(y[0])  # Print the first signal value


# In[ ]:




