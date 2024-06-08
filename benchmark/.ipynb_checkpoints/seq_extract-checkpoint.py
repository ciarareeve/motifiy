import numpy as np

# Load validation data
print("Loading validation data...")
X_val = np.load('../data/processed/X.npy', allow_pickle=True)  # Ensure to use the correct file path
print(f"Validation data loaded. Shape: {X_val.shape}")

# Define the number of sequences to process
num_sequences_to_process = 100

# Extract the first 100 sequences
sequences_subset = X_val[:num_sequences_to_process]

# Define a function to convert one-hot encoded sequences back to nucleotide sequences
def one_hot_decode(one_hot_seq):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    return ''.join([mapping[np.argmax(pos)] for pos in one_hot_seq])

# Save the sequences in FASTA format
fasta_filename = 'sequences_subset.fa'
with open(fasta_filename, 'w') as fasta_file:
    for i, seq in enumerate(sequences_subset):
        decoded_seq = one_hot_decode(seq)
        fasta_file.write(f'>sequence_{i+1}\n')
        fasta_file.write(f'{decoded_seq}\n')

print(f"Saved {num_sequences_to_process} sequences to {fasta_filename}")



# If you are using genomic sequences, use findMotifsGenome.pl
# findMotifsGenome.pl ../results/sequences_subset.fa hg38 ../results/homer_output/ -len 8,10,12 -norevopp

# If you are using general sequences (not tied to a specific genome), use findMotifs.pl
# findMotifs.pl ../results/sequences_subset.fa fasta ../results/homer_output/ -len 8,10,12

