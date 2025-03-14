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
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer for adversarial training in Aim Focuser
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer wrapper module
    """

    def __init__(self, alpha=1.0):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class ResidualBlock1D(nn.Module):
    """
    1D Residual block for feature extraction
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    1D ResNet for extracting learnable features from inertial signals
    """

    def __init__(self, in_channels=6, num_blocks=[2, 2, 2, 2], feature_dim=512):
        super(ResNet1D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, feature_dim)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock1D(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [batch_size, channels, seq_len]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class InterpretableFeatureExtractor(nn.Module):
    """
    Extracts interpretable features from inertial signals based on statistical principles
    """

    def __init__(self, feature_dim=512):
        super(InterpretableFeatureExtractor, self).__init__()
        # Statistical features -> feature_dim
        self.projection = nn.Linear(30, feature_dim)  # 30 statistical features

    def forward(self, x):
        # Extract interpretable statistical features
        # x shape: [batch_size, channels, seq_len]
        batch_size = x.shape[0]
        channels = x.shape[1]

        # Calculate statistical features
        mean_vals = torch.mean(x, dim=2)  # Mean along time
        std_vals = torch.std(x, dim=2)  # Standard deviation
        max_vals = torch.max(x, dim=2)[0]  # Maximum value
        min_vals = torch.min(x, dim=2)[0]  # Minimum value

        # Zero crossing rate
        zero_cross = torch.sum((x[:, :, :-1] * x[:, :, 1:]) < 0, dim=2).float()

        # Calculate statistical features within frequency domain
        # Using FFT magnitudes
        x_fft = torch.abs(torch.fft.rfft(x, dim=2))
        fft_mean = torch.mean(x_fft, dim=2)
        fft_std = torch.std(x_fft, dim=2)
        fft_max = torch.max(x_fft, dim=2)[0]

        # Autocorrelation features
        x_centered = x - x.mean(dim=2, keepdim=True)
        autocorr = torch.sum(x_centered * torch.roll(x_centered, shifts=1, dims=2), dim=2) / torch.sum(x_centered ** 2,
                                                                                                       dim=2)

        # Combine all features
        features = torch.cat([
            mean_vals, std_vals, max_vals, min_vals, zero_cross,
            fft_mean, fft_std, fft_max, autocorr
        ], dim=1)  # shape: [batch_size, 9*channels]

        # Project to the desired feature dimension
        interpretable_features = self.projection(features)

        return interpretable_features


class AimFocuser(nn.Module):
    """
    Aim Focuser module for filtering out content-specific information
    """

    def __init__(self, feature_dim=512, num_id_classes=10, num_char_classes=62, beta=1.0):
        super(AimFocuser, self).__init__()
        self.feature_dim = feature_dim
        self.beta = beta  # Information bottleneck trade-off parameter

        # Identity classifier - focuses on writer identification
        self.id_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_id_classes)
        )

        # Character classifier - with gradient reversal for adversarial training
        self.grl = GradientReversal(alpha=1.0)
        self.char_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_char_classes)
        )

    def forward(self, learnable_features, interpretable_features):
        # Combine learnable and interpretable features
        combined_features = torch.cat([learnable_features, interpretable_features], dim=1)

        # Identity classification branch
        id_logits = self.id_classifier(combined_features)

        # Character classification branch with gradient reversal
        reversed_features = self.grl(combined_features)
        char_logits = self.char_classifier(reversed_features)

        return id_logits, char_logits, combined_features

    def information_bottleneck_loss(self, features, labels):
        """
        Calculate information bottleneck loss to compress feature space
        """
        # Estimate feature entropy using KDE
        batch_size = features.size(0)
        # Use euclidean distance to compute pairwise distances
        distances = torch.cdist(features, features, p=2)

        # Set bandwidth for KDE (use median heuristic)
        bandwidth = 0.5 * torch.median(distances.view(-1))
        if bandwidth == 0:  # Avoid zero bandwidth
            bandwidth = 0.1

        # Compute kernel matrix using RBF kernel
        kernel_matrix = torch.exp(-distances / (2 * bandwidth ** 2))

        # Diagonal elements should be 1 (self-similarity)
        kernel_matrix.fill_diagonal_(1.0)

        # Estimate entropy of features (H(F))
        entropy_features = -torch.log(torch.mean(kernel_matrix))

        # Estimate conditional entropy H(Y|F) via cross-entropy loss
        # This is approximated by the cross-entropy loss of the identity classifier

        # Return the IB loss: H(F) - β*I(F;Y) = H(F) - β*(H(Y) - H(Y|F))
        # Since H(Y) is constant w.r.t. the parameters, we use H(F) - β*(-H(Y|F))
        return entropy_features - self.beta * (-entropy_features)


class RenyiEntropyRegularization(nn.Module):
    """
    Rényi-entropy-based Representation Regularization for category features
    """

    def __init__(self, feature_dim=512, num_categories=10, alpha=2):
        super(RenyiEntropyRegularization, self).__init__()
        self.feature_dim = feature_dim
        self.num_categories = num_categories
        self.alpha = alpha  # Rényi entropy order parameter

        # Initialize category encoding matrix E
        self.category_encoding = nn.Parameter(torch.randn(self.feature_dim, self.num_categories))
        nn.init.xavier_normal_(self.category_encoding)

    def forward(self, id_logits):
        # Compute category features using classification probabilities
        id_probs = F.softmax(id_logits, dim=1)  # shape: [batch_size, num_categories]
        category_features = torch.matmul(id_probs, self.category_encoding.t())  # shape: [batch_size, feature_dim]

        return category_features

    def compute_regularization(self):
        """
        Compute Rényi entropy regularization for category encoding matrix
        """
        # Compute Gram matrix G
        G = torch.matmul(self.category_encoding.t(), self.category_encoding)  # shape: [num_categories, num_categories]

        # Normalize G to G_tilde
        G_diag = torch.diag(G)
        G_tilde = torch.zeros_like(G)

        for i in range(self.num_categories):
            for j in range(self.num_categories):
                G_tilde[i, j] = G[i, j] / torch.sqrt(G_diag[i] * G_diag[j]) / 16.0

        # Trace normalization
        G_tilde = G_tilde / torch.trace(G_tilde)

        # Compute Rényi entropy
        if self.alpha == 1:
            # Special case: Shannon entropy (limit as alpha->1)
            eigenvalues = torch.linalg.eigvalsh(G_tilde)
            eigenvalues = eigenvalues[eigenvalues > 0]  # Keep only positive eigenvalues
            entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues))
        else:
            # General case: Rényi entropy
            G_tilde_power = torch.matrix_power(G_tilde, self.alpha)
            trace_powered = torch.trace(G_tilde_power)
            entropy = 1.0 / (1.0 - self.alpha) * torch.log2(trace_powered)

        # Return regularization term (inverse of entropy)
        return 1.0 / entropy


class TransformerAggregator(nn.Module):
    """
    Transformer for feature interaction and aggregation
    """

    def __init__(self, feature_dim=512, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerAggregator, self).__init__()
        self.feature_dim = feature_dim

        # Positional encoding for transformer
        self.pos_encoding = nn.Parameter(torch.zeros(1, 3, feature_dim))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=4 * feature_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(feature_dim * 3, feature_dim)

    def forward(self, learnable_features, interpretable_features, category_features):
        # Prepare input for transformer: [batch_size, 3, feature_dim]
        # Each sample has 3 feature vectors: learnable, interpretable, and category
        batch_size = learnable_features.shape[0]

        # Stack features along sequence dimension
        features = torch.stack([
            learnable_features,
            interpretable_features,
            category_features
        ], dim=1)  # shape: [batch_size, 3, feature_dim]

        # Add positional encoding
        features = features + self.pos_encoding

        # Pass through transformer encoder
        transformed_features = self.transformer_encoder(features)  # shape: [batch_size, 3, feature_dim]

        # Reshape and project to final output representation
        transformed_features = transformed_features.reshape(batch_size, -1)  # shape: [batch_size, 3*feature_dim]
        output_features = self.output_projection(transformed_features)  # shape: [batch_size, feature_dim]

        return output_features


class FGZSL(nn.Module):
    """
    Feature-Guided Zero-Shot Learning for handwriting verification
    """

    def __init__(self, input_channels=6, feature_dim=512, num_id_classes=10, num_char_classes=62, ib_beta=1.0):
        super(FGZSL, self).__init__()
        self.feature_dim = feature_dim

        # Feature extractors
        self.resnet = ResNet1D(in_channels=input_channels, feature_dim=feature_dim)
        self.interpretable_extractor = InterpretableFeatureExtractor(feature_dim=feature_dim)

        # Aim Focuser
        self.aim_focuser = AimFocuser(
            feature_dim=feature_dim,
            num_id_classes=num_id_classes,
            num_char_classes=num_char_classes,
            beta=ib_beta
        )

        # Rényi-entropy-based Representation Regularization
        self.rrr = RenyiEntropyRegularization(
            feature_dim=feature_dim,
            num_categories=num_id_classes,
            alpha=2
        )

        # Transformer for feature interaction and aggregation
        self.transformer = TransformerAggregator(
            feature_dim=feature_dim,
            num_heads=4,
            num_layers=2,
            dropout=0.1
        )

    def forward(self, x, inference=False):
        # Extract learnable features using ResNet
        learnable_features = self.resnet(x)

        # Extract interpretable features
        interpretable_features = self.interpretable_extractor(x)

        # Apply Aim Focuser to get filtered features
        id_logits, char_logits, combined_features = self.aim_focuser(
            learnable_features, interpretable_features
        )

        # Generate category features using RRR
        category_features = self.rrr(id_logits)

        # Aggregate features using transformer
        comprehensive_features = self.transformer(
            learnable_features,
            interpretable_features,
            category_features
        )

        if inference:
            return comprehensive_features

        return comprehensive_features, id_logits, char_logits, combined_features

    def compute_losses(self, id_logits, char_logits, combined_features, id_labels, char_labels):
        # Identity classification loss
        id_loss = F.cross_entropy(id_logits, id_labels)

        # Character classification loss (adversarial)
        char_loss = F.cross_entropy(char_logits, char_labels)

        # Information bottleneck loss
        ib_loss = self.aim_focuser.information_bottleneck_loss(combined_features, id_labels)

        # Rényi-entropy regularization for category features
        renyi_reg = self.rrr.compute_regularization()

        # Total loss
        total_loss = id_loss + char_loss + ib_loss + renyi_reg

        return total_loss, id_loss, char_loss, ib_loss, renyi_reg

    def verify_identity(self, x1, x2):
        """
        Verify if two handwriting samples belong to the same person
        """
        # Extract comprehensive features for both samples
        features1 = self.forward(x1, inference=True)
        features2 = self.forward(x2, inference=True)

        # Calculate cosine similarity
        similarity = F.cosine_similarity(features1, features2, dim=1)

        # Use a threshold to determine if they're from the same person
        # Threshold can be tuned based on validation data
        threshold = 0.85
        predictions = (similarity > threshold).float()

        return predictions, similarity


class InertialsDataset(Dataset):
    """
    Dataset class for inertial sensor data
    """

    def __init__(self, data, id_labels, char_labels):
        self.data = data
        self.id_labels = id_labels
        self.char_labels = char_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.id_labels[idx], self.char_labels[idx]


class VerificationDataset(Dataset):
    """
    Dataset class for verification (pairs of samples)
    """

    def __init__(self, data1, data2, labels):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx], self.labels[idx]


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=100, patience=10):
    """
    Train the FGZSL model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (data, id_labels, char_labels) in enumerate(train_loader):
            data, id_labels, char_labels = data.to(device), id_labels.to(device), char_labels.to(device)

            # Forward pass
            comprehensive_features, id_logits, char_logits, combined_features = model(data)

            # Compute losses
            total_loss, id_loss, char_loss, ib_loss, renyi_reg = model.compute_losses(
                id_logits, char_logits, combined_features, id_labels, char_labels
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Gradient clipping
            optimizer.step()

            train_loss += total_loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {total_loss.item():.4f}, ID Loss: {id_loss.item():.4f}, "
                      f"Char Loss: {char_loss.item():.4f}, IB Loss: {ib_loss.item():.4f}, "
                      f"Rényi Reg: {renyi_reg.item():.4f}")

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, id_labels, char_labels in val_loader:
                data, id_labels, char_labels = data.to(device), id_labels.to(device), char_labels.to(device)

                # Forward pass
                comprehensive_features, id_logits, char_logits, combined_features = model(data)

                # Compute losses
                total_loss, _, _, _, _ = model.compute_losses(
                    id_logits, char_logits, combined_features, id_labels, char_labels
                )

                val_loss += total_loss.item()

        val_loss /= len(val_loader)

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_verification(model, test_dataset, batch_size=64):
    """
    Evaluate the model on verification task
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    all_preds = []
    all_similarities = []
    all_labels = []

    with torch.no_grad():
        for data1, data2, labels in test_loader:
            data1, data2 = data1.to(device), data2.to(device)

            # Get verification predictions
            preds, similarities = model.verify_identity(data1, data2)

            all_preds.extend(preds.cpu().numpy())
            all_similarities.extend(similarities.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)

    # Calculate precision, recall, F1 score
    true_positives = np.sum((all_preds == 1) & (all_labels == 1))
    false_positives = np.sum((all_preds == 1) & (all_labels == 0))
    false_negatives = np.sum((all_preds == 0) & (all_labels == 1))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(
        f"Verification Results: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'predictions': all_preds,
        'similarities': all_similarities,
        'labels': all_labels
    }

    return results