
# Motify

Motify is a comprehensive tool for motif finding and analysis in DNA sequences. It identifies motifs, generates consensus sequences, compares them against known motifs, and provides visual representations of motifs.

## Features

- **Motif Identification:** Find motifs in DNA sequences from direct input or file uploads (FASTA, VCF).
- **Consensus Motif:** Generate the consensus sequence representing the most common pattern.
- **Position Weight Matrix (PWM):** Create a PWM to represent nucleotide frequencies at each position.
- **Motif Logo:** Generate visual representations of motifs using `logomaker`.
- **User Interface:** Access the tool via a user-friendly web interface or a command-line interface (CLI).

## Installation

To install Motify, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/ciarareeve/motify.git
   cd motify
   ```

2. Set up a conda environment:

   ```bash
   conda create --name motif_finder_env python=3.9
   conda activate motif_finder_env
   ```

3. Install the required dependencies:

   ```bash
   pip install -e .
   ```

## Usage

### Web Interface

To run the web application, use the following command:

```bash
motif_finder --web
```

Open your web browser and go to `http://127.0.0.1:5000/`. You can input a DNA sequence or upload a file to find motifs and view results directly in the browser.

### Command-Line Interface (CLI)

To run Motify from the command line, use the following commands:

#### Using a Sequence

```bash
motif_finder --sequence "ATGCTAGCTAG" --out result.html
```

#### Using a File

```bash
motif_finder --file path/to/your/file.fasta --out result.html
```

### Example Dataset

A minimal test example dataset is provided in the repository as `test_seq.fasta`. You can use this file to test the functionality of Motify.

#### Example FASTA File (`test_seq.fasta`)

```
>Sequence1
ATGCTAGCTAG
>Sequence2
CGTACGTAGC
>Sequence3
TTAGGCTAAC
```

## Project Structure

```
motifiy/
├── README.md
├── main.py
├── ml_model
│   └── model.py
├── motif_finder
│   ├── known_motifs.json
│   ├── motif.py
│   ├── static
│   │   └── motif_logo.png
│   └── test_seq.fasta
├── requirements.txt
├── setup.py
└── web_app
    ├── app.py
    └── templates
        ├── index.html
        └── result.html

```
GitHub Repository: [Motify GitHub Repository](https://github.com/ciarareeve/motifiy)

