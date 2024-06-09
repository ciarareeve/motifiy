Motify
======

Motify is a tool for identifying and fine-mapping sequence motifs from ChIP-seq data using deep learning. By leveraging the power of BPNet and Integrated Gradients, Motify provides an advanced approach to motif discovery that offers finer resolution compared to traditional methods.

Features
--------

* **Sequence Scanning:** Scans input sequences to detect motifs.
* **Fine Mapping:** Uses Integrated Gradients to attribute contributions to specific nucleotide positions, allowing for fine mapping of motifs.
* **Contextual Detection:** Identifies motifs in their genomic context.
* **Visualization:** Generates visual representations of motif attributions and clustering results.

Why Use Motify for Motif Finding?
---------------------------------

Traditional methods for motif finding, such as peak calling followed by Position Weight Matrix (PWM) generation, often provide a broad overview of potential binding sites but can lack the resolution needed to understand the precise sequence features driving these bindings. Motify leverages deep learning techniques to predict ChIP-seq readouts directly from raw sequence data, which allows for a more nuanced and fine-grained analysis. By using BPNet for prediction and Integrated Gradients for feature attribution, Motify identifies not just the presence of motifs, but also their specific contributions to the ChIP-seq signal within their genomic context. This approach can reveal subtle sequence variations and dependencies that traditional methods might miss, providing a more comprehensive and detailed map of regulatory elements. Additionally, the fine-mapping capability of Motify can improve our understanding of motif functionality and interactions, offering insights into complex gene regulation mechanisms. This makes Motify a powerful tool for researchers aiming to uncover the intricacies of genomic regulation with higher precision.

* **High Resolution:** Offers finer resolution compared to traditional Position Weight Matrices (PWMs).
* **Interpretability:** Uses Integrated Gradients for feature attribution, making it easier to understand the model's predictions.
* **Versatility:** Can be used with both histone and transcription factor ChIP-seq data.

Installation
------------

**Prerequisites**

Ensure you have the following installed:

* Python 3.10 or later
* Anaconda or Miniconda (recommended for managing dependencies)

**Install Instructions for macOS and Windows**

Clone the repository:

.. code-block:: bash

    git clone https://github.com/ciarareeve/motify.git
    cd motify

Create a virtual environment:

.. code-block:: bash

    conda create -n motify_env python=3.10
    conda activate motify_env

Install dependencies:

.. code-block:: bash

    pip install -r requirements.txt

Download required data: Place your reference genome (e.g., hg38.fa) and ChIP-seq data (e.g., chip_seq_data.bed) in the `data/raw` directory.

**Additional Steps for Windows Users**

* Install Visual Studio Build Tools: Download and install Visual Studio Build Tools from here.
* Ensure Long Paths are Enabled: Enable long path support on Windows by editing the registry or using the Group Policy Editor.

Usage
-----

**Step 1: Prepare Data**

Ensure your reference genome and ChIP-seq data are placed in the `data/raw` directory.

.. code-block:: bash

    python scripts/data_preparation.py

**Step 2: Train the Model**

Run the model training script:

.. code-block:: bash

    python scripts/mlm_training.py

**Step 3: Compute Attributions with Integrated Gradients**

After the model has finished training, run the Integrated Gradients attribution script:

.. code-block:: bash

    python scripts/attributions.py

**Step 4: Cluster Seqlets**

Finally, run the clustering script:

.. code-block:: bash

    python scripts/clustering.py

**Before Proceeding:**

Please note that a pre-trained model is already available in the `results` folder. This allows you to start directly from Step 4. Training the model is time-intensive; therefore, for tutorial purposes, we provide a pre-trained model based on histone mark H3K27ac data with the hg38 reference genome. You can fine-tune this model and all subsequent scripts to suit your specific application needs. Feel free to run the provided Jupyter Notebook scripts for more interactive tuning.

Script Explanations
--------------------

**data_preparation.py**

*Objective:* Process ChIP-seq data to extract sequences from the reference genome.

*Process:*
- Load the reference genome.
- Load and filter ChIP-seq data.
- Extract sequences from the genome.
- Save the processed sequences and signal values.

**mlm_training.py**

*Objective:* Train a BPNet-like model to predict ChIP-seq readouts.

*Process:*
- Load and preprocess the data.
- Split the data into training and validation sets.
- Create and compile the BPNet-like model.
- Train the model with early stopping and model checkpointing.
- Save the trained model and training history.

**attributions.py**

*Objective:* Compute attributions for the trained model using Integrated Gradients.

*Process:*
- Load the trained BPNet model.
- One-hot encode the validation data.
- Compute attributions for a subset of the validation data.
- Save the computed attributions.
- Visualize the attributions for the first sequence.

**clustering.py**

*Objective:* Extract high-attribution seqlets and cluster them.

*Process:*
- Load the attributions and input data.
- Extract high-attribution seqlets.
- Cluster the seqlets using DBSCAN.
- Save the clustering results.
- Visualize the clusters.

Requirements
------------

* numpy==1.23.5
* pandas==1.5.3
* scikit-learn==1.2.2
* tensorflow==2.16.1
* matplotlib==3.6.2
* pyfaidx==0.6.0.1
* IPython==8.9.0

Contact
-------

For any questions or issues, please contact cireeve@ucsd.edu.

