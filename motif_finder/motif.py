import json
import pandas as pd
from Bio.Seq import Seq
from Bio import motifs
from collections import Counter
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logomaker
from Bio import SeqIO

def load_known_motifs(filepath='known_motifs.json'):
    """
    Load known motifs from a JSON file.

    Args:
        filepath (str): The path to the JSON file containing the known motifs.

    Returns:
        list: A list of tuples, where each tuple contains the name of the motif and the motif object.
    """
    with open(filepath, 'r') as file:
        data = json.load(file)
    known_motifs = []
    for entry in data['motifs']:
        motif_seq = Seq(entry['sequence'])
        m = motifs.create([motif_seq])
        known_motifs.append((entry['name'], m))
    return known_motifs

def find_motifs(sequences):
    """
    Find motifs in a list of sequences.

    Args:
        sequences (list): A list of DNA sequences.

    Returns:
        tuple: A tuple containing the consensus sequence and the motif object.
    """
    instances = [Seq(seq) for seq in sequences]
    m = motifs.create(instances)
    consensus = m.consensus
    return consensus, m

def compare_motifs(found_motif, known_motifs):
    """
    Compare a found motif to a list of known motifs.

    Args:
        found_motif (str): The consensus sequence of the found motif.
        known_motifs (list): A list of tuples, where each tuple contains the name of the motif and the motif object.

    Returns:
        str: The name of the known motif if a match is found, otherwise "Unknown".
    """
    for name, motif in known_motifs:
        if str(motif.consensus) == str(found_motif):
            return name
    return "Unknown"

def generate_background_sequences(sequences, num_background=100):
    """
    Generate background sequences for a given list of sequences.

    Args:
        sequences (list): A list of DNA sequences.
        num_background (int): The number of background sequences to generate.

    Returns:
        list: A list of randomly generated background sequences.
    """
    background_sequences = []
    for seq in sequences:
        bg_seq = ''.join(random.choices('ACGT', k=len(seq)))
        background_sequences.append(bg_seq)
    return background_sequences

def visualize_motif(pwm, output_file='motif_logo.png'):
    """
    Visualize a motif using a position weight matrix (PWM).

    Args:
        pwm (list): A list of lists representing the position weight matrix.
        output_file (str): The path to save the motif logo image.

    Returns:
        None
    """
    pwm_df = pd.DataFrame(pwm)
    pwm_df = pwm_df.rename(columns={0: 'A', 1: 'C', 2: 'G', 3: 'T'})
    pwm_df.index.name = 'Position'
    logo = logomaker.Logo(pwm_df)
    plt.savefig(output_file)

def convert_to_kmer_frequency(sequence, k=6):
    """
    Convert a DNA sequence to k-mer frequency.

    Args:
        sequence (str): A DNA sequence.
        k (int): The length of the k-mer.

    Returns:
        dict: A dictionary where the keys are k-mers and the values are their frequencies.
    """
    kmer_counts = Counter([sequence[i:i+k] for i in range(len(sequence) - k + 1)])
    total_kmers = sum(kmer_counts.values())
    kmer_freqs = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}
    return kmer_freqs

def read_sequences_from_file(filepath):
    """
    Read DNA sequences from a file.

    Args:
        filepath (str): The path to the file containing the DNA sequences.

    Returns:
        list: A list of DNA sequences.
    """
    sequences = []
    file_format = filepath.split('.')[-1].lower()
    if file_format in ['fa', 'fasta', 'fna']:
        for record in SeqIO.parse(filepath, "fasta"):
            sequences.append(str(record.seq))
    elif file_format in ['vcf']:
        for record in SeqIO.parse(filepath, "vcf"):
            sequences.append(str(record.seq))
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    return sequences
