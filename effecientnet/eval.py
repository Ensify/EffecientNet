import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch model.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model file.")
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the dataset.")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument('--img-height', type=int, default=384, help="Height of input images.")
    parser.add_argument('--img-width', type=int, default=384, help="Width of input images.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(args.dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model = models.efficientnet_v2_s(weights=None, num_classes=len(dataset.classes))
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    # Evaluation
    criterion = nn.CrossEntropyLoss()
    val_loss = 0
    correct = 0
    total = 0
    losses = []
    accuracies = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Track loss and accuracy
            losses.append(loss.item())
            accuracies.append((predicted == labels).sum().item() / labels.size(0))

    avg_loss = val_loss / len(dataloader)
    accuracy = 100 * correct / total
    print(f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Plot loss and accuracy
    plt.figure()
    plt.plot(losses, label='Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Batch')
    plt.legend()
    plt.savefig('loss_plot.png')

    plt.figure()
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Batch')
    plt.legend()
    plt.savefig('accuracy_plot.png')

if __name__ == "__main__":
    main()
