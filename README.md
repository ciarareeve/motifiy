
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
   git clone https://github.com/ciarareeve/motify.git
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
   Place your reference genome (e.g., `hg38.fa`) and ChIP-seq data (e.g., `chip_seq_data.bed`) in the `data/raw` directory.

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
