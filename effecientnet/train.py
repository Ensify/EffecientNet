import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PyTorch model.")
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the dataset.")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--img-height', type=int, default=384, help="Height of input images.")
    parser.add_argument('--img-width', type=int, default=384, help="Width of input images.")
    parser.add_argument('--validation-split', type=float, default=0.2, help="Fraction of data to use for validation.")
    parser.add_argument('--num-epochs', type=int, default=1, help="Number of epochs to train.")
    parser.add_argument('--learning-rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--experiment-name', type=str, required=True, help="Name of the experiment.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Create directory for the experiment
    experiment_dir = os.path.join('experiments', args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(args.dataset_path,"train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.dataset_path,"val"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model setup
    model = models.efficientnet_v2_s(weights=None, num_classes=len(train_dataset.classes)).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize metrics lists
    train_losses = []
    val_losses = []
    accuracies = []

    best_accuracy = 0

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        val_losses.append(avg_val_loss)
        accuracies.append(accuracy)

        print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save the model checkpoint if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(experiment_dir, 'model_best.pth'))

        # Save the last model checkpoint
        torch.save(model.state_dict(), os.path.join(experiment_dir, 'model_last.pth'))

    # Plot training metrics
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, 'training_loss_plot.png'))

    plt.figure()
    plt.plot(accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, 'validation_accuracy_plot.png'))

if __name__ == "__main__":
    main()
