from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from motif_finder.motif import convert_to_kmer_frequency  # Correct import

class SequenceClassifier:
    """
    A class for training and predicting sequence classifications using k-mer frequencies.

    Attributes:
        clf (RandomForestClassifier): Random Forest classifier for sequence classification.
        kmer_list (list): List of k-mers present in the training sequences.

    Methods:
        train(sequences, labels): Trains the classifier using the given sequences and labels.
        predict(sequence): Predicts the class label for the given sequence.
        convert_to_kmer_vector(sequence, k): Converts a sequence into a k-mer frequency vector.
        build_kmer_list(sequences, k): Builds the list of k-mers present in the training sequences.
    """

    def __init__(self):
        """
        Initializes the SequenceClassifier object with a Random Forest classifier.
        """
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, sequences, labels):
        """
        Trains the classifier using the given sequences and labels.

        Args:
            sequences (list): List of input sequences.
            labels (list): List of corresponding class labels.
        """
        X = [self.convert_to_kmer_vector(seq) for seq in sequences]
        y = labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.clf.fit(X_train, y_train)
        print(f"Training accuracy: {self.clf.score(X_train, y_train)}")
        print(f"Testing accuracy: {self.clf.score(X_test, y_test)}")

    def predict(self, sequence):
        """
        Predicts the class label for the given sequence.

        Args:
            sequence (str): Input sequence.

        Returns:
            int: Predicted class label.
        """
        X = self.convert_to_kmer_vector(sequence)
        return self.clf.predict([X])

    def convert_to_kmer_vector(self, sequence, k=6):
        """
        Converts a sequence into a k-mer frequency vector.

        Args:
            sequence (str): Input sequence.
            k (int): Length of k-mers. Default is 6.

        Returns:
            list: K-mer frequency vector.
        """
        kmer_freqs = convert_to_kmer_frequency(sequence, k)
        kmer_vector = [kmer_freqs.get(kmer, 0) for kmer in self.kmer_list]
        return kmer_vector

    def build_kmer_list(self, sequences, k=6):
        """
        Builds the list of k-mers present in the training sequences.

        Args:
            sequences (list): List of input sequences.
            k (int): Length of k-mers. Default is 6.
        """
        kmer_set = set()
        for sequence in sequences:
            kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
            kmer_set.update(kmers)
        self.kmer_list = sorted(list(kmer_set))

# Example usage:
# clf = SequenceClassifier()
# sequences = ["ATGCTAGCTAG", "GCTAGCTACG", "TACGATCGTA"]
# labels = [0, 1, 1]
# clf.build_kmer_list(sequences)
# clf.train(sequences, labels)
# print(clf.predict("ATGCTAGCTA"))