import numpy as np
import matplotlib.pyplot as plt
import logomaker
import pandas as pd
import os

# Create directories to save images if they don't exist
images_dir = '../results/images'
os.makedirs(images_dir, exist_ok=True)

# Load attributions
print("Loading attributions...")
attributions = np.load('../results/sequences/attributions_subset.npy') 
print(f"Attributions shape: {attributions.shape}")

# Create function to generate sequence logo DataFrame
def create_sequence_logo(attributions, length):
    logo_df = pd.DataFrame(attributions[:length], columns=['A', 'C', 'G', 'T'])
    return logo_df

# Plot normalized attributions and sequence logos for each sequence
# this is the sample number of 25 sequences chosen at random (for tutorial purposes)
og_seq = [41906, 7297, 1640, 48599, 18025, 16050, 14629, 9145,48266, 6718,44349, 48541, 35742, 5698, 38699, 27652, 2083, 1953, 6141, 14329, 15248, 33119, 39454, 1740, 36782]


n = attributions.shape[0]


# Iterate over each sequence and create the plot
for i in range(og_seq):
    norm_attributions = attributions[i] / np.max(np.abs(attributions[i]))
    
    # Plot normalized attributions
    plt.figure(figsize=(10, 6))
    plt.plot(norm_attributions[:400])  # Plot only the first 400 positions for better visibility
    plt.title(f'Integrated Gradients Attributions for Original Sequence {og_seq[i]} (Normalized)')
    plt.xlabel('Position')
    plt.ylabel('Normalized Attribution Score')
    plt.savefig(os.path.join(images_dir, f'attributions_sequence_{og_seq[i]}.png'))
    plt.show()


print("Plots saved.")

