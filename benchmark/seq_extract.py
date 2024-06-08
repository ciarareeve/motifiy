import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# Load the dataset
X_val = np.load('../data/processed/X.npy', allow_pickle=True)
print(f"Validation data loaded. Shape: {X_val.shape}")

# Determine the length of each sequence
sequence_lengths = [len(seq) for seq in X_val]

# Print some statistics about the sequence lengths
print(f"Total number of sequences: {len(sequence_lengths)}")
print(f"Minimum sequence length: {min(sequence_lengths)}")
print(f"Maximum sequence length: {max(sequence_lengths)}")
print(f"Average sequence length: {np.mean(sequence_lengths)}")
print(f"Median sequence length: {np.median(sequence_lengths)}")

# Select a diverse subset of sequences
random.seed(42)  # For reproducibility
indices = random.sample(range(len(X_val)), 100)
diverse_sequences = [X_val[i] for i in indices]

# One-hot encode the selected sequences
def one_hot_encode(sequence, max_len):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    one_hot = np.zeros((max_len, 4), dtype=np.int8)
    for i, char in enumerate(sequence[:max_len]):
        if char in mapping:
            one_hot[i, mapping[char]] = 1
    return one_hot

max_len = max(sequence_lengths)
diverse_sequences_encoded = [one_hot_encode(seq, max_len) for seq in diverse_sequences]

# Save the selected sequences in FASTA format
=======

# Load validation data
print("Loading validation data...")
X_val = np.load('../data/processed/X.npy', allow_pickle=True)  # Ensure to use the correct file path
print(f"Validation data loaded. Shape: {X_val.shape}")

# Define the number of sequences to process
num_sequences_to_process = 100

# Extract the first 100 sequences
sequences_subset = X_val[:num_sequences_to_process]

# Define a function to convert one-hot encoded sequences back to nucleotide sequences
>>>>>>> ea235ad8debe94649d5edfd32961ae39931b64ad
def one_hot_decode(one_hot_seq):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    return ''.join([mapping[np.argmax(pos)] for pos in one_hot_seq])

<<<<<<< HEAD
fasta_filename = 'diverse_sequences_subset.fa'
with open(fasta_filename, 'w') as fasta_file:
    for i, seq in enumerate(diverse_sequences_encoded):
=======
# Save the sequences in FASTA format
fasta_filename = 'sequences_subset.fa'
with open(fasta_filename, 'w') as fasta_file:
    for i, seq in enumerate(sequences_subset):
>>>>>>> ea235ad8debe94649d5edfd32961ae39931b64ad
        decoded_seq = one_hot_decode(seq)
        fasta_file.write(f'>sequence_{i+1}\n')
        fasta_file.write(f'{decoded_seq}\n')

<<<<<<< HEAD
print(f"Saved {len(diverse_sequences)} diverse sequences to {fasta_filename}")
=======
print(f"Saved {num_sequences_to_process} sequences to {fasta_filename}")



# If you are using genomic sequences, use findMotifsGenome.pl
# findMotifsGenome.pl ../results/sequences_subset.fa hg38 ../results/homer_output/ -len 8,10,12 -norevopp

# If you are using general sequences (not tied to a specific genome), use findMotifs.pl
# findMotifs.pl ../results/sequences_subset.fa fasta ../results/homer_output/ -len 8,10,12
# Motify

Motify is a tool for identifying and fine-mapping sequence motifs from ChIP-seq data using deep learning. By leveraging the power of BPNet and Integrated Gradients, Motify provides an advanced approach to motif discovery that offers finer resolution compared to traditional methods.

## Features

- **Sequence Scanning**: Scans input sequences to detect motifs.
- **Fine Mapping**: Uses Integrated Gradients to attribute contributions to specific nucleotide positions, allowing for fine mapping of motifs.
- **Contextual Detection**: Identifies motifs in their genomic context.
- **Visualization**: Generates visual representations of motif attributions and clustering results.

### Why Use Motify for Motif Finding?

Traditional methods for motif finding, such as peak calling followed by Position Weight Matrix (PWM) generation, often provide a broad overview of potential binding sites but can lack the resolution needed to understand the precise sequence features driving these bindings. Motify leverages deep learning techniques to predict ChIP-seq readouts directly from raw sequence data, which allows for a more nuanced and fine-grained analysis. By using BPNet for prediction and Integrated Gradients for feature attribution, Motify identifies not just the presence of motifs, but also their specific contributions to the ChIP-seq signal within their genomic context. This approach can reveal subtle sequence variations and dependencies that traditional methods might miss, providing a more comprehensive and detailed map of regulatory elements. Additionally, the fine-mapping capability of Motify can improve our understanding of motif functionality and interactions, offering insights into complex gene regulation mechanisms. This makes Motify a powerful tool for researchers aiming to uncover the intricacies of genomic regulation with higher precision.

- **High Resolution**: Offers finer resolution compared to traditional Position Weight Matrices (PWMs).
- **Interpretability**: Uses Integrated Gradients for feature attribution, making it easier to understand the model's predictions.
- **Versatility**: Can be used with both histone and transcription factor ChIP-seq data.

## Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.10 or later
- Anaconda or Miniconda (recommended for managing dependencies)

### Install Instructions for macOS and Windows

1. **Clone the repository**:
   ```sh
   git clone https://github.com/ciara/motify.git
   cd motify
   ```

2. **Create a virtual environment**:
   ```sh
   conda create -n motify_env python=3.10
   conda activate motify_env
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Download required data**:
   Place your reference genome (e.g., `hg38.fa`) and ChIP-seq data (e.g., `chseq2.bed`) in the `data/raw` directory.

### Additional Steps for Windows Users

1. **Install Visual Studio Build Tools**:
   Download and install Visual Studio Build Tools from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

2. **Ensure Long Paths are Enabled**:
   Enable long path support on Windows by editing the registry or using the Group Policy Editor.

## Usage

### Step 1: Prepare Data

Ensure your reference genome and ChIP-seq data are placed in the `data/raw` directory.

### Step 2: Process Data

Run the data processing script to prepare your sequences:

```sh
python scripts/data_processing.py
```

### Step 3: Train the Model

Run the model training script:

```sh
python scripts/model_training.py
```

### Step 4: Compute Attributions with Integrated Gradients

After the model has finished training, run the Integrated Gradients attribution script:

```sh
python scripts/integrated_gradients_attribution.py
```

### Step 5: Cluster Seqlets

Finally, run the clustering script:

```sh
python scripts/clustering.py
```

## Script Explanations

### `data_processing.py`

- **Objective**: Process ChIP-seq data to extract sequences from the reference genome.
- **Process**:
  1. Load the reference genome.
  2. Load and filter ChIP-seq data.
  3. Extract sequences from the genome.
  4. Save the processed sequences and signal values.

### `model_training.py`

- **Objective**: Train a BPNet-like model to predict ChIP-seq readouts.
- **Process**:
  1. Load and preprocess the data.
  2. Split the data into training and validation sets.
  3. Create and compile the BPNet-like model.
  4. Train the model with early stopping and model checkpointing.
  5. Save the trained model and training history.

### `attributions.py`

- **Objective**: Compute attributions for the trained model using Integrated Gradients.
- **Process**:
  1. Load the trained BPNet model.
  2. One-hot encode the validation data.
  3. Compute attributions for a subset of the validation data.
  4. Save the computed attributions.
  5. Visualize the attributions for the first sequence.

### `clustering.py`

- **Objective**: Extract high-attribution seqlets and cluster them.
- **Process**:
  1. Load the attributions and input data.
  2. Extract high-attribution seqlets.
  3. Cluster the seqlets using DBSCAN.
  4. Save the clustering results.
  5. Visualize the clusters.

## Requirements

```
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
tensorflow==2.16.1
matplotlib==3.6.2
pyfaidx==0.6.0.1
IPython==8.9.0
```

## Contact

For any questions or issues, please contact cireeve@ucsd.edu.

