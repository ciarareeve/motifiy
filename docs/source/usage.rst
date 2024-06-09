Usage
=====

Motify is a tool for identifying and fine-mapping sequence motifs from ChIP-seq data using deep learning.

Installation
------------

Clone the repository and install dependencies:

.. code-block:: bash

    git clone https://github.com/ciarareeve/motify.git
    cd motify
    conda create -n motify_env python=3.10
    conda activate motify_env
    pip install -r requirements.txt

Ensure your reference genome and ChIP-seq data are placed in the `data/raw` directory.

Data Preparation
----------------

Run the data preparation script:

.. code-block:: bash

    python scripts/data_preparation.py

Model Training
--------------

Run the model training script:

.. code-block:: bash

    python scripts/mlm_training.py

Compute Attributions
--------------------

Run the Integrated Gradients attribution script:

.. code-block:: bash

    python scripts/attributions.py

Cluster Seqlets
---------------

Run the clustering script:

.. code-block:: bash

    python scripts/clustering.py

Before Proceeding
-----------------

A pre-trained model is already available in the `results` folder. This allows you to start directly from the attribution step. Training the model is time-intensive; therefore, for tutorial purposes, we provide a pre-trained model based on histone mark H3K27ac data with the hg38 reference genome. You can fine-tune this model and all subsequent scripts to suit your specific application needs. Feel free to run the provided Jupyter Notebook scripts for more interactive tuning.

Contact
-------

For any questions or issues, please contact cireeve@ucsd.edu.

