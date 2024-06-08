import numpy as np
import pandas as pd
from pyfaidx import Fasta

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
=======
def load_genome(reference_genome_path):
    return Fasta(reference_genome_path)

def load_chip_seq_data(chip_seq_path):
    chip_seq_data = pd.read_csv(chip_seq_path, sep='\t', header=None)
    chip_seq_data.columns = ['chrom', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak']
    return chip_seq_data

def extract_sequences(chip_seq_data, genome):
    sequences = []
    for _, row in chip_seq_data.iterrows():
        sequence = genome[row['chrom']][row['start']:row['end']].seq
        sequences.append(sequence)
    return np.array(sequences)

def main():
    reference_genome_path = 'data/raw/hg19.fa'
    chip_seq_path = 'data/raw/chip_seq_data.bed'

    genome = load_genome(reference_genome_path)
    chip_seq_data = load_chip_seq_data(chip_seq_path)
    sequences = extract_sequences(chip_seq_data, genome)

    np.save('data/processed/X.npy', sequences)
    np.save('data/processed/y.npy', chip_seq_data['signalValue'].values)

if __name__ == "__main__":
    main()


