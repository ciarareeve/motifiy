# scripts/clustering.py

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

def extract_seqlets(attributions, threshold=0.5):
    seqlets = []
    for attribution in attributions:
        high_attribution_indices = np.where(attribution > threshold)[0]
        for index in high_attribution_indices:
            seqlet = attribution[max(0, index-10):min(len(attribution), index+10)]
            seqlets.append(seqlet)
    return np.array(seqlets)

def cluster_seqlets(seqlets, eps=0.5, min_samples=5):
    similarity_matrix = cosine_similarity(seqlets)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(1 - similarity_matrix)
    return clustering.labels_

def fine_grained_clustering(seqlets, labels):
    unique_labels = np.unique(labels)
    clusters = {}
    for label in unique_labels:
        if label == -1:
            continue
        cluster_seqlets = [seqlets[i] for i in range(len(seqlets)) if labels[i] == label]
        clusters[label] = cluster_seqlets

    fine_grained_clusters = {}
    for label, cluster_seqlets in clusters.items():
        similarity_matrix = cosine_similarity(cluster_seqlets)
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='precomputed').fit(1 - similarity_matrix)
        fine_grained_clusters[label] = clustering.labels_
    
    return fine_grained_clusters

def main():
    attributions = np.load('results/attributions/attributions.npy')
    seqlets = extract_seqlets(attributions)

    positive_seqlets = [seqlet for seqlet in seqlets if np.mean(seqlet) > 0]
    negative_seqlets = [seqlet for seqlet in seqlets if np.mean(seqlet) < 0]

    positive_labels = cluster_seqlets(positive_seqlets)
    negative_labels = cluster_seqlets(negative_seqlets)

    positive_fine_clusters = fine_grained_clustering(positive_seqlets, positive_labels)
    negative_fine_clusters = fine_grained_clustering(negative_seqlets, negative_labels)

    np.save('results/clusters/positive_clusters.npy', positive_fine_clusters)
    np.save('results/clusters/negative_clusters.npy', negative_fine_clusters)

if __name__ == "__main__":
    main()

