import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import numpy as np

def train(args):
    # Create experiment directory
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    train_dataset = datasets.ImageFolder(root=os.path.join(args.dataset_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(args.dataset_dir, 'val'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Print class indices
    print("Class indices:", train_dataset.class_to_idx)
    input("Verify class weights and press enter to train...")

    # Model
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Adjust for binary classification
    model = model.to(args.device)

    # Loss function with class weights
    class_weights = torch.tensor(args.class_weights, dtype=torch.float).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training Loop
    num_epochs = args.epochs
    train_losses = []
    val_losses = []
    val_precisions = []
    val_recalls = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        # Calculate precision and recall
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        val_precisions.append(np.mean(precision))
        val_recalls.append(np.mean(recall))

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Precision: {val_precisions[-1]:.4f}, Recall: {val_recalls[-1]:.4f}')

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(args.experiment_dir, 'best_model.pth'))

        # Save the last model
        torch.save(model.state_dict(), os.path.join(args.experiment_dir, 'last_model.pth'))

    # Plot training and validation losses
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, marker='o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.experiment_dir, 'training_validation_loss.png'))

    # Plot precision and recall
    plt.figure()
    plt.plot(range(1, num_epochs + 1), val_precisions, marker='o', label='Validation Precision')
    plt.plot(range(1, num_epochs + 1), val_recalls, marker='o', label='Validation Recall')
    plt.title('Validation Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(os.path.join(args.experiment_dir, 'precision_recall.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a MobileNetV2 model for binary classification.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory.')
    parser.add_argument('--experiment_dir', type=str, required=True, help='Directory to save experiment results.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training.')
    parser.add_argument('--class_weights', type=float, nargs=2, required=True, help='Class weights for loss function.')
    args = parser.parse_args()

    train(args)
