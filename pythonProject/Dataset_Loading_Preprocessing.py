import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from Framework import *
import time

def preprocess_inertial_data(data_path, resample_length=400):
    """
    Load and preprocess inertial sensor data
    """
    # This is a placeholder for actual data loading
    # In a real implementation, you would load data from files

    # Simulate loading data
    # For real implementation, replace with actual data loading code
    # Data shape: [num_samples, channels, seq_length]
    # Channels: 3 accelerometer axes + 3 gyroscope axes = 6 channels

    # Placeholder data
    num_samples = 1000
    num_channels = 6  # 3 accelerometer + 3 gyroscope
    seq_length = resample_length

    # Create random data for demonstration
    data = np.random.randn(num_samples, num_channels, seq_length)

    # Create random labels
    id_labels = np.random.randint(0, 10, size=num_samples)  # 10 different writers
    char_labels = np.random.randint(0, 62, size=num_samples)  # 62 different characters

    # Convert to torch tensors
    data = torch.FloatTensor(data)
    id_labels = torch.LongTensor(id_labels)
    char_labels = torch.LongTensor(char_labels)

    return data, id_labels, char_labels


def create_verification_pairs(data, id_labels, num_pairs=10000, positive_ratio=0.5):
    """
    Create verification pairs (same/different writer)
    """
    num_samples = len(data)

    # Calculate number of positive and negative pairs
    num_positive = int(num_pairs * positive_ratio)
    num_negative = num_pairs - num_positive

    # Initialize arrays
    data1 = []
    data2 = []
    pair_labels = []

    # Create positive pairs (same writer)
    for _ in range(num_positive):
        writer_idx = np.random.randint(0, 10)  # Pick a random writer
        # Find samples from this writer
        writer_samples = np.where(id_labels == writer_idx)[0]
        if len(writer_samples) < 2:
            continue  # Skip if not enough samples

        # Sample two different samples from the same writer
        idx1, idx2 = np.random.choice(writer_samples, size=2, replace=False)
        data1.append(data[idx1])
        data2.append(data[idx2])
        pair_labels.append(1)  # Same writer

    # Create negative pairs (different writers)
    for _ in range(num_negative):
        # Pick two different writers
        writer1_idx = np.random.randint(0, 10)
        writer2_idx = np.random.randint(0, 10)
        while writer2_idx == writer1_idx:
            writer2_idx = np.random.randint(0, 10)

        # Find samples from these writers
        writer1_samples = np.where(id_labels == writer1_idx)[0]
        writer2_samples = np.where(id_labels == writer2_idx)[0]

        if len(writer1_samples) == 0 or len(writer2_samples) == 0:
            continue  # Skip if not enough samples

        # Sample one sample from each writer
        idx1 = np.random.choice(writer1_samples)
        idx2 = np.random.choice(writer2_samples)

        data1.append(data[idx1])
        data2.append(data[idx2])
        pair_labels.append(0)  # Different writers

    # Convert to tensors
    data1 = torch.stack(data1)
    data2 = torch.stack(data2)
    pair_labels = torch.LongTensor(pair_labels)

    return data1, data2, pair_labels


def prepare_datasets(data_path, train_ratio=0.8, val_ratio=0.1):
    """
    Prepare datasets for training and evaluation
    """
    # Load and preprocess data
    data, id_labels, char_labels = preprocess_inertial_data(data_path)

    # Split data for training, validation, and testing
    num_samples = len(data)
    indices = np.arange(num_samples)

    # Split indices
    train_indices, temp_indices = train_test_split(
        indices, train_size=train_ratio, stratify=id_labels.numpy()
    )

    # Calculate validation size relative to the remaining data
    val_size = val_ratio / (1 - train_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices, train_size=val_size, stratify=id_labels[temp_indices].numpy()
    )

    # Create datasets
    train_dataset = InertialsDataset(
        data[train_indices], id_labels[train_indices], char_labels[train_indices]
    )

    val_dataset = InertialsDataset(
        data[val_indices], id_labels[val_indices], char_labels[val_indices]
    )

    test_dataset = InertialsDataset(
        data[test_indices], id_labels[test_indices], char_labels[test_indices]
    )

    # Create verification pairs for testing
    test_data1, test_data2, test_pair_labels = create_verification_pairs(
        data[test_indices], id_labels[test_indices], num_pairs=1000
    )

    test_verification_dataset = VerificationDataset(test_data1, test_data2, test_pair_labels)

    # Create unseen category verification dataset for the ZSL task
    # Normally, this would be loaded from separate files containing unseen writers' data
    # Here we're just creating random data as a placeholder
    unseen_data, unseen_id_labels, unseen_char_labels = preprocess_inertial_data(data_path + "_unseen")

    unseen_data1, unseen_data2, unseen_pair_labels = create_verification_pairs(
        unseen_data, unseen_id_labels, num_pairs=1000
    )

    unseen_verification_dataset = VerificationDataset(unseen_data1, unseen_data2, unseen_pair_labels)

    return train_dataset, val_dataset, test_verification_dataset, unseen_verification_dataset