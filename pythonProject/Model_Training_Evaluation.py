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

def main():
    # Set parameters
    feature_dim = 512
    num_id_classes = 10  # Number of writers in training set
    num_char_classes = 62  # Number of characters (10 digits + 52 uppercase/lowercase letters)
    batch_size = 64
    num_epochs = 100
    patience = 10
    learning_rate = 3e-4
    weight_decay = 5e-3

    # Prepare datasets
    data_path = "path/to/your/data"  # Replace with actual path
    train_dataset, val_dataset, test_verification_dataset, unseen_verification_dataset = prepare_datasets(data_path)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = FGZSL(
        input_channels=6,
        feature_dim=feature_dim,
        num_id_classes=num_id_classes,
        num_char_classes=num_char_classes,
        ib_beta=1.0
    )

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate / 100
    )

    # Train the model
    model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        num_epochs=num_epochs,
        patience=patience
    )

    # Evaluate on seen categories
    print("Evaluating on seen categories...")
    seen_results = evaluate_verification(model, test_verification_dataset, batch_size=batch_size)

    # Evaluate on unseen categories
    print("Evaluating on unseen categories...")
    unseen_results = evaluate_verification(model, unseen_verification_dataset, batch_size=batch_size)

    # Save model
    torch.save(model.state_dict(), "fgzsl_model.pth")

    print("Training and evaluation completed.")

    # Measure inference time
    measure_inference_time(model)


def measure_inference_time(model):
    """
    Measure inference time of each module in the FGZSL framework
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create random input data
    input_data = torch.randn(1, 6, 400).to(device)  # [batch_size, channels, seq_length]

    # Warm up
    for _ in range(10):
        _ = model(input_data, inference=True)

    # Measure time for each module
    with torch.no_grad():
        # Full inference
        start_time = time.time()
        for _ in range(100):
            _ = model(input_data, inference=True)
        full_time = (time.time() - start_time) / 100 * 1000  # ms

        # ResNet (learnable features)
        start_time = time.time()
        for _ in range(100):
            _ = model.resnet(input_data)
        resnet_time = (time.time() - start_time) / 100 * 1000  # ms

        # Interpretable feature extractor
        start_time = time.time()
        for _ in range(100):
            _ = model.interpretable_extractor(input_data)
        interpretable_time = (time.time() - start_time) / 100 * 1000  # ms

        # Aim Focuser (identity classifier only during inference)
        learnable_features = model.resnet(input_data)
        interpretable_features = model.interpretable_extractor(input_data)
        start_time = time.time()
        for _ in range(100):
            _ = model.aim_focuser.id_classifier(torch.cat([learnable_features, interpretable_features], dim=1))
        aim_focuser_time = (time.time() - start_time) / 100 * 1000  # ms

        # Transformer
        id_logits = model.aim_focuser.id_classifier(torch.cat([learnable_features, interpretable_features], dim=1))
        category_features = model.rrr(id_logits)
        start_time = time.time()
        for _ in range(100):
            _ = model.transformer(learnable_features, interpretable_features, category_features)
        transformer_time = (time.time() - start_time) / 100 * 1000  # ms

    # Print results
    print("\nInference Time Breakdown:")
    print(f"Total Inference Time: {full_time:.2f} ms")
    print(f"ResNet (Learnable Features): {resnet_time:.2f} ms")
    print(f"Interpretable Feature Extractor: {interpretable_time:.2f} ms")
    print(f"Aim Focuser (Inference mode): {aim_focuser_time:.2f} ms")
    print(f"Transformer Aggregator: {transformer_time:.2f} ms")

    # Calculate other times
    other_time = full_time - (resnet_time + interpretable_time + aim_focuser_time + transformer_time)
    print(f"Other Operations: {other_time:.2f} ms")


def visualize_features(model, data_loader, num_samples=500):
    """
    Visualize feature distributions using t-SNE
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Extract features and labels
    features = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, id_label, _) in enumerate(data_loader):
            if len(features) >= num_samples:
                break

            data = data.to(device)
            batch_features = model(data, inference=True).cpu().numpy()

            features.append(batch_features)
            labels.append(id_label.numpy())

    # Concatenate features and labels
    features = np.vstack(features)
    labels = np.concatenate(labels)

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', s=50, alpha=0.8)
    plt.colorbar(scatter, label='Writer ID')
    plt.title('t-SNE Visualization of Feature Distributions')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('feature_visualization.png')
    plt.close()


if __name__ == "__main__":
    main()