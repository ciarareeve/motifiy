# scripts/data_preparation.py

import numpy as np
import pandas as pd
from pyfaidx import Fasta

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

