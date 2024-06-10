# Motify

[![Documentation Status](https://readthedocs.org/projects/motify/badge/?version=latest)](https://motify.readthedocs.io/en/latest/?badge=latest)

Visit our homepage here https://motify.readthedocs.io

Motify is a tool for identifying and fine-mapping sequence motifs from ChIP-seq data using deep learning. By leveraging the power of BPNet and Integrated Gradients, Motify provides an advanced approach to motif discovery that offers finer resolution compared to traditional methods.

## Features

- **Sequence Scanning:** Scans input sequences to detect motifs.
- **Fine Mapping:** Uses Integrated Gradients to attribute contributions to specific nucleotide positions, allowing for fine mapping of motifs.
- **Contextual Detection:** Identifies motifs in their genomic context.
- **Visualization:** Generates visual representations of motif attributions and clustering results.

## Why Use Motify for Motif Finding?

Traditional methods for motif finding, such as peak calling followed by Position Weight Matrix (PWM) generation, often provide a broad overview of potential binding sites but can lack the resolution needed to understand the precise sequence features driving these bindings. Motify leverages deep learning techniques to predict ChIP-seq readouts directly from raw sequence data, which allows for a more nuanced and fine-grained analysis. By using BPNet for prediction and Integrated Gradients for feature attribution, Motify identifies not just the presence of motifs, but also their specific contributions to the ChIP-seq signal within their genomic context. This approach can reveal subtle sequence variations and dependencies that traditional methods might miss, providing a more comprehensive and detailed map of regulatory elements. Additionally, the fine-mapping capability of Motify can improve our understanding of motif functionality and interactions, offering insights into complex gene regulation mechanisms. This makes Motify a powerful tool for researchers aiming to uncover the intricacies of genomic regulation with higher precision.

- **High Resolution:** Offers finer resolution compared to traditional Position Weight Matrices (PWMs).
- **Interpretability:** Uses Integrated Gradients for feature attribution, making it easier to understand the model's predictions.
- **Versatility:** Can be used with both histone and transcription factor ChIP-seq data.

## Before Proceeding:
Please note that a pre-trained model is already available in the `results` folder. This allows you to start directly from Step 4. Training the model is time-intensive; therefore, for tutorial purposes, we provide a pre-trained model trained on histone mark H3K27ac data with the hg38 reference genome. You can fine-tune this model and all subsequent scripts to suit your specific application needs. Feel free to run the provided Jupyter Notebook scripts for more interactive tuning.


## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.10 or later
- Anaconda or Miniconda (recommended for managing dependencies)

### Install Instructions for macOS and Windows

1. Clone the repository:

   ```bash
   git clone https://github.com/ciarareeve/motify.git
   cd motify
   ```

2. Create a virtual environment:

   ```bash
   conda create -n motify_env python=3.10
   conda activate motify_env
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Instructions for Downloading and Setting Up the Necessary Files


To get the necessary data files, download the latest release and follow the instructions from [here](https://github.com/ciarareeve/motify/releases/tag/v1.0.0).


### Included Files in Release:
- **X.npy.xz**: Processed numpy array containing one-hot encoded sequences used as input for the model.
- **y.npy**: Processed numpy array containing labels or output values corresponding to the input sequences in `X.npy`.
- **hg19.fa.gz**: FASTA file of the hg19 reference genome.
- **hg38.fa.gz**: FASTA file of the hg38 reference genome.

### Steps:

1. **Download the Files from the GitHub Release**:
    - Download the following files:
        - `X.npy.xz` (mandatory for tutorial run)
        - `y.npy`(mandatory for tutorial run)
        - `hg19.fa.gz`
        - `hg38.fa.gz`(mandatory for tutorial run)

2. **Move the Files to the Appropriate Directories**:
    - Move `X.npy.xz` and `y.npy` to the `data/processed` directory.
    - Move `hg19.fa.gz` and `hg38.fa.gz` to the `data/raw` directory.

    Example commands:
    ```bash
    mv X.npy.xz data/processed/
    mv y.npy data/processed/
    mv hg19.fa.gz data/raw/
    mv hg38.fa.gz data/raw/
    ```

3. **Uncompress the Files**:

    - Uncompress `X.npy.xz`:
        ```bash
        xz -d data/processed/X.npy.xz
        ```

    - Uncompress `hg19.fa.gz`:
        ```bash
        gunzip data/raw/hg19.fa.gz
        ```

    - Uncompress `hg38.fa.gz`:
        ```bash
        gunzip data/raw/hg38.fa.gz
        ```

4. **Verify the File Structure**:
    Ensure that the files are in the correct directories with the following structure:
    ```
    ├── data
    │   ├── processed
    │   │   ├── X.npy
    │   │   └── y.npy
    │   └── raw
    │       ├── hg19.fa
    │       └── hg38.fa
    ```

By following these steps you are now ready to run the Motify tool.

Feel free to reach out if you encounter any issues or have any questions.


### Additional Steps for Windows Users

- Install Visual Studio Build Tools: Download and install Visual Studio Build Tools from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
- Ensure Long Paths are Enabled: Enable long path support on Windows by editing the registry or using the Group Policy Editor.

## Usage

### Step 1: Prepare Data

Ensure your reference genome and ChIP-seq data are placed in the `data/raw` directory.

### Step 2: Process Data

Run the data processing script to prepare your sequences:

```bash
python scripts/data_processing.py
```

### Step 3: Train the Model

Run the model training script:

```bash
python scripts/model_training.py
```

### Step 4: Compute Attributions with Integrated Gradients

After the model has finished training, run the Integrated Gradients attribution script:

```bash
python scripts/integrated_gradients_attribution.py
```

### Step 5: Cluster Seqlets

Finally, run the clustering script:

```bash
python scripts/clustering.py
```

### Step 6: Generate HTML Report

Run the report generation script to create a visual report of attributions and clusters:

```bash
python scripts/report.py
```

## Script Explanations

### data_processing.py

**Objective:** Process ChIP-seq data to extract sequences from the reference genome.

**Process:**
- Load the reference genome.
- Load and filter ChIP-seq data.
- Extract sequences from the genome.
- Save the processed sequences and signal values.

### mlm_training.py

**Objective:** Train a BPNet-like model to predict ChIP-seq readouts.

**Process:**
- Load and preprocess the data.
- Split the data into training and validation sets.
- Create and compile the BPNet-like model.
- Train the model with early stopping and model checkpointing.
- Save the trained model and training history.

### integrated_gradients_attribution.py

**Objective:** Compute attributions for the trained model using Integrated Gradients.

**Process:**
- Load the trained BPNet model.
- One-hot encode the validation data.
- Compute attributions for a subset of the validation data.
- Save the computed attributions.
- Visualize the attributions for the selected sequences.

### clustering.py

**Objective:** Extract high-attribution seqlets and cluster them.

**Process:**
- Load the attributions and input data.
- Extract high-attribution seqlets.
- Cluster the seqlets using DBSCAN.
- Save the clustering results.
- Visualize the clusters.

### report.py

**Objective:** Generate an HTML report to visualize attributions and sequence logos.

**Process:**
- Generate HTML for attributions images.
- Generate HTML for cluster images.
- Replace placeholders in the template HTML file with generated HTML.
- Write the final HTML to a new file.

## Requirements

- numpy==1.23.5
- pandas==1.5.3
- scikit-learn==1.2.2
- tensorflow==2.16.1
- matplotlib==3.6.2
- pyfaidx==0.6.0.1
- IPython==8.9.0
- logomaker==0.8

## Contact

For any questions or issues, please contact [cireeve@ucsd.edu](mailto:cireeve@ucsd.edu).
