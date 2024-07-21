import argparse
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, class_names, file_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(file_path)
    plt.close()

def evaluate(args):
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    val_dataset = datasets.ImageFolder(root=os.path.join(args.dataset_dir), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = models.efficientnet_v2_s(weights=None, num_classes = len(val_dataset.classes))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Adjust for binary classification
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)
    model.eval()

    # Evaluate
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute precision and recall
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    #Compute F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f'F1 Score: {f1_score:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)

    # Plot and save confusion matrix
    class_names = ['Defect', 'Normal']  # Adjust as per your dataset class names
    plot_confusion_matrix(cm, class_names, os.path.join(args.experiment_dir, 'confusion_matrix.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate an EfficientNet model for binary classification.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model weights.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for evaluation.')
    parser.add_argument('--experiment_dir', type=str, required=True, help='Directory to save evaluation results.')
    args = parser.parse_args()

    if not args.experiment_dir:
        raise ValueError('You must provide --experiment_dir.')

    # Ensure experiment directory exists
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    evaluate(args)

#python effecientnet\eval.py --dataset_dir ..\DataProcess\datasets\classification\defect_classification\val --model_path effnet_1.pth --batch_size 4 --experiment_dir effnet_2