### Motify README

# Motify

Motify is a web-based and command-line tool designed to find and visualize DNA sequence motifs. The tool allows users to input DNA sequences and identify matching motifs based on predefined motif files. It then generates visual representations of these motifs using probability weight matrices (PWMs).

## Features

- **Web Application**: An intuitive web interface for inputting DNA sequences and viewing matching motifs.
- **Command-Line Interface (CLI)**: A robust CLI for processing sequences and generating HTML reports of matching motifs.
- **Motif Visualization**: Generates logos for motifs using PWMs, providing a clear visual representation.
- **Expandable**: Supports degenerate base sequences, allowing for a comprehensive search of motif variations.

## Installation

To set up and run Motify, follow these steps:

### Requirements

Motify requires Python 3.6 or higher. Ensure you have Python installed on your system.

### Using requirements.txt

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/motify.git
    cd motify
    ```

2. Create a virtual environment and activate it:
   ```bash
   conda create --name motify python=3.8  # or your preferred Python version
   conda activate motify
   conda install --file requirements.txt
   ```

## Configuration

1. **Motif Files**: Place your `.motif` files in the `motifs` directory inside the project folder. This directory should contain the motif files used by Motify to find matching motifs in the input sequences.

## Running the Application

### Web Application

To run the web application, use the following command:

```bash
python main.py --web
```

The web app will start, and you can access it via `http://localhost:5000` in your web browser.

### Command-Line Interface

To run the CLI, use the following commands:

- **Find motifs in a sequence**:
    ```bash
    python main.py --sequence ACTGACTGACTG --out result.html
    ```

- **Find motifs in a file**:
    ```bash
    python main.py --file sequences.txt --out result.html
    ```

- **Run in verbose mode**:
    ```bash
    python main.py --sequence ACTGACTGACTG --out result.html --verbose
    ```

## Usage

### Web Application

1. Open your web browser and navigate to `http://localhost:5000`.
2. Enter your DNA sequence in the provided input field.
3. Click the submit button to process the sequence and find matching motifs.
4. View the results on the next page, which will display the matching motifs and their logos.

### Command-Line Interface

1. Use the `--sequence` flag to input a sequence directly or the `--file` flag to input a file containing sequences.
2. The `--out` flag specifies the output HTML file where the results will be saved.
3. The `--verbose` flag enables detailed logging for debugging purposes.

## Example

Here's an example of how to use the CLI:

```bash
python main.py --sequence ACTGACTGACTG --out result.html --verbose
```

This command will process the input sequence `ACTGACTGACTG`, find matching motifs, generate logos, and save the results to `result.html`.


## Setup Script

Create a `setup.py` file with the following contents:

```python
from setuptools import setup, find_packages

setup(
    name='motify',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Flask==2.0.2',
        'pandas==1.3.3',
        'logomaker==0.8',
        'matplotlib==3.4.3'
    ],
    entry_points={
        'console_scripts': [
            'motify=main:main',
        ],
    },
)
```

## Contributing

We welcome contributions! Please fork the repository and submit pull requests.


## Contact

For any questions or inquiries, please contact cireeve@ucsd.edu

