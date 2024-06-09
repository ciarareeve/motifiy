import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import logomaker
import pandas as pd
import os

# Create directories to save seqlets and images if they don't exist
seqlets_dir = '../results/seqlets'
os.makedirs(seqlets_dir, exist_ok=True)
images_dir = '../results/images'
os.makedirs(images_dir, exist_ok=True)

# Load attributions and input data
print("Loading attributions and input data...")
attributions = np.load('../results/sequences/attributions_subset.npy')  # Ensure the correct file path
input_data = np.load('../data/processed/X.npy', allow_pickle=True)  # Ensure the correct file path
print(f"Attributions shape: {attributions.shape}")
print(f"Input data shape: {input_data.shape}")

# Define the one-hot encoding function
def one_hot_encode(sequence, max_len):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    one_hot = np.zeros((max_len, 4), dtype=np.int8)
    for i, char in enumerate(sequence[:max_len]):
        if char in mapping:
            one_hot[i, mapping[char]] = 1
    return one_hot

# Define the maximum sequence length
max_len = 18593

# One-hot encode input data if necessary
print("One-hot encoding input data...")
input_data_encoded = np.array([one_hot_encode(seq, max_len) for seq in input_data])
print(f"One-hot encoded input data. Shape: {input_data_encoded.shape}")

# Extract high-attribution seqlets with a lower threshold
def extract_high_attribution_seqlets(attributions, input_data, threshold=0.015):  # Adjusted threshold
    seqlets = []
    for seq_idx, seq_attributions in enumerate(attributions):
        high_attr_positions = np.where(np.max(np.abs(seq_attributions), axis=1) > threshold)[0]
        for pos in high_attr_positions:
            if pos-10 >= 0 and pos+10 < input_data.shape[1]:  # Ensure seqlet is within bounds
                seqlet = input_data[seq_idx, pos-10:pos+10, :]  # Example: 20bp seqlets
                seqlets.append(seqlet)
    return np.array(seqlets)

print("Extracting high-attribution seqlets...")
high_attribution_seqlets = extract_high_attribution_seqlets(attributions, input_data_encoded)
print(f"Total high-attribution seqlets extracted: {len(high_attribution_seqlets)}")

# Filter out low-variance seqlets
def filter_low_variance_seqlets(seqlets, variance_threshold=0.001):
    variances = np.var(seqlets, axis=(1, 2))
    high_variance_seqlets = seqlets[variances > variance_threshold]
    return high_variance_seqlets

high_attribution_seqlets = filter_low_variance_seqlets(high_attribution_seqlets)
print(f"High-attribution seqlets after filtering: {len(high_attribution_seqlets)}")

# Save seqlets
np.save(os.path.join(seqlets_dir, 'high_attribution_seqlets.npy'), high_attribution_seqlets)
print("High-attribution seqlets saved.")

# Cluster seqlets using DBSCAN
def cluster_seqlets(seqlets, eps=0.3, min_samples=3):
    seqlet_vectors = seqlets.reshape(len(seqlets), -1)
    similarity_matrix = cosine_similarity(seqlet_vectors)
    distance_matrix = 1 - similarity_matrix
    distance_matrix[distance_matrix < 0] = 0  # Ensure no negative values
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance_matrix)
    return clustering.labels_

print("Clustering high-attribution seqlets...")
labels = cluster_seqlets(high_attribution_seqlets)
print(f"Clustering completed. Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")

# Visualize clusters
print("Visualizing clusters...")
plt.figure(figsize=(10, 6))
plt.scatter(range(len(labels)), labels, c=labels, cmap='viridis', s=5)
plt.colorbar()
plt.xlabel('Seqlet Index')
plt.ylabel('Cluster Label')
plt.title('Seqlet Clustering')
plt.savefig(os.path.join(images_dir, 'seqlet_clustering.png'))
plt.show()
print("Clusters visualization completed.")

# Save clustering results
np.save('../results/clusters/seqlet_clusters.npy', labels)
print("Clustering results saved.")

#Create and visualize sequence logos for each cluster
unique_labels = set(labels)
for cluster in unique_labels:
    if cluster != -1:  # Ignore noise points labeled as -1
        cluster_seqlets = high_attribution_seqlets[labels == cluster]
        avg_attributions = np.mean(cluster_seqlets, axis=0)
        
        # Create sequence logo
        plot_length = min(400, avg_attributions.shape[0])  # Adjust the plot length as needed
        logo_df = pd.DataFrame(avg_attributions[:plot_length], columns=['A', 'C', 'G', 'T'])
        
        # Plot sequence logo
        plt.figure(figsize=(20, 4))
        logomaker.Logo(logo_df, color_scheme='classic')
        plt.title(f'Sequence Logo for Cluster {cluster}')
        plt.savefig(os.path.join(images_dir, f'sequence_logo_cluster_{cluster}.png'))
        plt.show()

print("Sequence logos created and saved.")




