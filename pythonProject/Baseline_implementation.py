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


class HMM_Verifier:
    """
    Hidden Markov Model for handwriting verification
    """

    def __init__(self, n_states=5, n_components=2):
        from hmmlearn import hmm

        self.n_states = n_states
        self.models = []

        # Create a HMM model for each writer
        for _ in range(10):  # 10 writers
            model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag')
            self.models.append(model)

    def fit(self, data, id_labels):
        # Reshape data for HMM
        data = data.numpy()
        id_labels = id_labels.numpy()

        # Train a separate HMM for each writer
        for writer_id in range(10):
            writer_indices = np.where(id_labels == writer_id)[0]
            writer_data = data[writer_indices]

            # Reshape data to 2D array [n_samples * seq_length, n_features]
            writer_data_flat = []
            for sample in writer_data:
                # Transpose to make sequence dimension first
                sample = sample.transpose(1, 0)  # Now: [seq_length, channels]
                writer_data_flat.append(sample)

            writer_data_flat = np.vstack(writer_data_flat)

            # Fit the model
            self.models[writer_id].fit(writer_data_flat)

    def verify(self, data1, data2):
        # Reshape data
        data1 = data1.numpy()
        data2 = data2.numpy()

        num_pairs = data1.shape[0]
        predictions = np.zeros(num_pairs)

        for i in range(num_pairs):
            # Get scores for both samples from all models
            sample1 = data1[i].transpose(1, 0)  # [seq_length, channels]
            sample2 = data2[i].transpose(1, 0)  # [seq_length, channels]

            best_model1_idx = None
            best_model1_score = -np.inf
            best_model2_idx = None
            best_model2_score = -np.inf

            for model_idx, model in enumerate(self.models):
                score1 = model.score(sample1)
                score2 = model.score(sample2)

                if score1 > best_model1_score:
                    best_model1_score = score1
                    best_model1_idx = model_idx

                if score2 > best_model2_score:
                    best_model2_score = score2
                    best_model2_idx = model_idx

            # If both samples are best described by the same model, they are likely from the same person
            predictions[i] = 1 if best_model1_idx == best_model2_idx else 0

        return predictions


class DTW_Verifier:
    """
    Dynamic Time Warping for handwriting verification (AirSign method)
    """

    def __init__(self):
        from scipy.spatial.distance import euclidean
        from fastdtw import fastdtw

        self.threshold = 15.0  # Threshold for determination
        self.features = None
        self.id_labels = None

    def extract_statistical_features(self, data):
        """
        Extract handcrafted statistical features
        """
        # data shape: [batch_size, channels, seq_length]
        batch_size = data.shape[0]
        features = []

        for i in range(batch_size):
            sample = data[i]  # [channels, seq_length]

            # Calculate statistical features for each channel
            sample_features = []

            for c in range(sample.shape[0]):
                channel = sample[c]  # [seq_length]

                # Time domain features
                mean = np.mean(channel)
                std = np.std(channel)
                maxval = np.max(channel)
                minval = np.min(channel)

                # Frequency domain features
                fft_vals = np.abs(np.fft.rfft(channel))
                fft_mean = np.mean(fft_vals)
                fft_std = np.std(fft_vals)

                # Combine features
                channel_features = [mean, std, maxval, minval, fft_mean, fft_std]
                sample_features.extend(channel_features)

            features.append(sample_features)

        return np.array(features)

    def fit(self, data, id_labels):
        # Extract features for all training data
        self.features = self.extract_statistical_features(data.numpy())
        self.id_labels = id_labels.numpy()

    def verify(self, data1, data2):
        # Extract features
        features1 = self.extract_statistical_features(data1.numpy())
        features2 = self.extract_statistical_features(data2.numpy())

        num_pairs = features1.shape[0]
        predictions = np.zeros(num_pairs)

        for i in range(num_pairs):
            # Calculate DTW distance
            distance, _ = fastdtw(features1[i], features2[i], dist=euclidean)

            # If distance is below threshold, they are from the same person
            predictions[i] = 1 if distance < self.threshold else 0

        return predictions


class CNNAuth_Verifier:
    """
    CNN + SVM for handwriting verification (CNNAuth method)
    """

    def __init__(self, input_channels=6):
        from sklearn.svm import SVC

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.AdaptiveAvgPool1d(1),
        )

        self.svm = SVC(kernel='rbf', probability=True)

    def extract_cnn_features(self, data):
        """
        Extract CNN features
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = self.cnn.to(device)

        features = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size].to(device)
                batch_features = self.cnn(batch).squeeze(-1).cpu().numpy()
                features.append(batch_features)

        return np.vstack(features)

    def fit(self, data, id_labels):
        # Extract CNN features
        cnn_features = self.extract_cnn_features(data)

        # Fit SVM classifier
        self.svm.fit(cnn_features, id_labels.numpy())

    def verify(self, data1, data2):
        # Extract CNN features
        features1 = self.extract_cnn_features(data1)
        features2 = self.extract_cnn_features(data2)

        # Predict writer IDs
        pred_ids1 = self.svm.predict(features1)
        pred_ids2 = self.svm.predict(features2)

        # If predicted IDs are the same, they are from the same person
        predictions = (pred_ids1 == pred_ids2).astype(int)

        return predictions


class InAirSignature_Verifier:
    """
    RNN-based verifier for handwriting (In-Air Signature method)
    """

    def __init__(self, input_channels=6, hidden_size=128):
        self.rnn = nn.GRU(input_channels, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 10)  # 10 writers
        self.threshold = 0.75  # Similarity threshold

    def fit(self, data, id_labels):
        """
        Train RNN model
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rnn = self.rnn.to(device)
        self.fc = self.fc.to(device)

        # Prepare data
        data = data.to(device)
        id_labels = id_labels.to(device)

        # Transpose to [batch_size, seq_length, channels]
        data = data.transpose(1, 2)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(list(self.rnn.parameters()) + list(self.fc.parameters()), lr=0.001)

        # Train the model
        num_epochs = 10
        batch_size = 64

        for epoch in range(num_epochs):
            # Create data loader
            dataset = torch.utils.data.TensorDataset(data, id_labels)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for batch_idx, (batch_data, batch_labels) in enumerate(data_loader):
                # Forward pass
                outputs, _ = self.rnn(batch_data)
                outputs = outputs[:, -1, :]  # Take the last time step
                logits = self.fc(outputs)

                # Compute loss
                loss = criterion(logits, batch_labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    def extract_features(self, data):
        """
        Extract RNN features
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rnn = self.rnn.to(device)

        # Transpose to [batch_size, seq_length, channels]
        data = data.to(device).transpose(1, 2)

        with torch.no_grad():
            outputs, _ = self.rnn(data)
            features = outputs[:, -1, :]  # Take the last time step

        return features.cpu()

    def verify(self, data1, data2):
        """
        Verify if two samples are from the same person
        """
        # Extract features
        features1 = self.extract_features(data1)
        features2 = self.extract_features(data2)

        # Calculate cosine similarity
        similarity = F.cosine_similarity(features1, features2, dim=1)

        # If similarity is above threshold, they are from the same person
        predictions = (similarity > self.threshold).int().numpy()

        return predictions


def compare_methods(data_path):
    """
    Compare different methods for handwriting verification
    """
    # Prepare datasets
    train_dataset, val_dataset, test_verification_dataset, unseen_verification_dataset = prepare_datasets(data_path)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Extract all training data for traditional methods
    all_train_data = []
    all_train_id_labels = []

    for data, id_labels, _ in train_loader:
        all_train_data.append(data)
        all_train_id_labels.append(id_labels)

    all_train_data = torch.cat(all_train_data, dim=0)
    all_train_id_labels = torch.cat(all_train_id_labels, dim=0)

    # Prepare test data
    test_data1, test_data2, test_labels = [], [], []
    for data1, data2, labels in DataLoader(test_verification_dataset, batch_size=64, shuffle=False):
        test_data1.append(data1)
        test_data2.append(data2)
        test_labels.append(labels)

    test_data1 = torch.cat(test_data1, dim=0)
    test_data2 = torch.cat(test_data2, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # Prepare unseen test data
    unseen_data1, unseen_data2, unseen_labels = [], [], []
    for data1, data2, labels in DataLoader(unseen_verification_dataset, batch_size=64, shuffle=False):
        unseen_data1.append(data1)
        unseen_data2.append(data2)
        unseen_labels.append(labels)

    unseen_data1 = torch.cat(unseen_data1, dim=0)
    unseen_data2 = torch.cat(unseen_data2, dim=0)
    unseen_labels = torch.cat(unseen_labels, dim=0)

    # Methods to compare
    methods = {
        "HMM": HMM_Verifier(),
        "AirSign (DTW)": DTW_Verifier(),
        "CNNAuth": CNNAuth_Verifier(),
        "In-Air Signature": InAirSignature_Verifier(),
        "FGZSL (ours)": FGZSL(
            input_channels=6,
            feature_dim=512,
            num_id_classes=10,
            num_char_classes=62,
            ib_beta=1.0
        )
    }

    # Train and evaluate each method
    results_seen = {}
    results_unseen = {}

    for name, method in methods.items():
        print(f"\nTraining {name}...")

        if name == "FGZSL (ours)":
            # For FGZSL, use the full training pipeline
            optimizer = torch.optim.AdamW(
                method.parameters(),
                lr=3e-4,
                weight_decay=5e-3
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=100,
                eta_min=3e-6
            )

            method = train_model(
                method,
                train_loader,
                DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4),
                optimizer,
                scheduler,
                num_epochs=50,
                patience=10
            )

            # Evaluate on seen categories
            print(f"Evaluating {name} on seen categories...")
            seen_results = evaluate_verification(method, test_verification_dataset)
            results_seen[name] = seen_results

            # Evaluate on unseen categories
            print(f"Evaluating {name} on unseen categories...")
            unseen_results = evaluate_verification(method, unseen_verification_dataset)
            results_unseen[name] = unseen_results
        else:
            # For other methods, use their specific training method
            method.fit(all_train_data, all_train_id_labels)

            # Evaluate on seen categories
            print(f"Evaluating {name} on seen categories...")
            predictions = method.verify(test_data1, test_data2)

            # Calculate metrics
            accuracy = np.mean(predictions == test_labels.numpy())
            true_positives = np.sum((predictions == 1) & (test_labels.numpy() == 1))
            false_positives = np.sum((predictions == 1) & (test_labels.numpy() == 0))
            false_negatives = np.sum((predictions == 0) & (test_labels.numpy() == 1))

            precision = true_positives / (true_positives + false_positives) if (
                                                                                           true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (
                                                                                        true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(
                f"Seen Results: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")

            results_seen[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'predictions': predictions,
                'labels': test_labels.numpy()
            }

            # Evaluate on unseen categories
            print(f"Evaluating {name} on unseen categories...")
            predictions = method.verify(unseen_data1, unseen_data2)

            # Calculate metrics
            accuracy = np.mean(predictions == unseen_labels.numpy())
            true_positives = np.sum((predictions == 1) & (unseen_labels.numpy() == 1))
            false_positives = np.sum((predictions == 1) & (unseen_labels.numpy() == 0))
            false_negatives = np.sum((predictions == 0) & (unseen_labels.numpy() == 1))

            precision = true_positives / (true_positives + false_positives) if (
                                                                                           true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (
                                                                                        true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(
                f"Unseen Results: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")

            results_unseen[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'predictions': predictions,
                'labels': unseen_labels.numpy()
            }

    # Print comparison tables
    print("\nComparison on Seen Categories:")
    print("Method\t\t\tAccuracy\tPrecision\tRecall\t\tF1-score")
    for name, result in results_seen.items():
        print(
            f"{name}\t\t{result['accuracy']:.4f}\t\t{result['precision']:.4f}\t\t{result['recall']:.4f}\t\t{result['f1_score']:.4f}")

    print("\nComparison on Unseen Categories:")
    print("Method\t\t\tAccuracy\tPrecision\tRecall\t\tF1-score")
    for name, result in results_unseen.items():
        print(
            f"{name}\t\t{result['accuracy']:.4f}\t\t{result['precision']:.4f}\t\t{result['recall']:.4f}\t\t{result['f1_score']:.4f}")