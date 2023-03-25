# Description: This script contains functions to preprocess the data for use in a neural network.
# Title: nn/preprocess.py
# Author: Cindy Pino-Barrios
# Date: 03/24/2023


# Imports libraries
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import pandas as pd

# Define the sample_seqs function

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Get the number of positive and negative labels
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos

    # Get the number of sequences to sample
    num_to_sample = min(num_pos, num_neg)

    # Create a list to store the sampled sequences
    sampled_seqs = []

    # Create a list to store the sampled labels

    sampled_labels = []

    # Iterate through the sequences and labels
    for seq, label in zip(seqs, labels):
        # If the label is positive and the number of positive samples is less than the number of samples to take
        if label and num_pos > 0:
            # Add the sequence and label to the sampled sequences and labels
            sampled_seqs.append(seq)
            sampled_labels.append(label)
            # Decrement the number of positive samples
            num_pos -= 1
        # If the label is negative and the number of negative samples is less than the number of samples to take
        elif not label and num_neg > 0:
            # Add the sequence and label to the sampled sequences and labels
            sampled_seqs.append(seq)
            sampled_labels.append(label)
            # Decrement the number of negative samples
            num_neg -= 1

    #assert len(sampled_seqs) == len(sampled_labels)

    # Return the sampled sequences and labels

      
    return sampled_seqs, sampled_labels



# Use the one_hot_encode_seqs function to one-hot encode the positive and negative sequences

# Use the one_hot_encode_seqs function to one-hot encode the positive and negative sequences

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # Create a dictionary to map each nucleotide to a one-hot encoding

    nucleotide_dict = {"A": [1, 0, 0, 0], 
                       "T": [0, 1, 0, 0], 
                       "C": [0, 0, 1, 0], 
                       "G": [0, 0, 0, 1]}

    # Create a list to store the one-hot encodings
    encodings = []
    
    # Loop through each sequence in the list of sequences

    for seq in seq_arr:
        # Create a list to store the one-hot encoding for the current sequence
        seq_encoding = []

        # Loop through each nucleotide in the sequence
        for nucleotide in seq:
            # Append the one-hot encoding for the current nucleotide to the current sequence encoding
            seq_encoding.extend(nucleotide_dict[nucleotide])

        # Append the current sequence encoding to the list of encodings
        encodings.append(seq_encoding)

    return np.array(encodings)
