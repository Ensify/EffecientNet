import argparse
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

def load_image_from_frame(frame, transform):
    """Load and transform a single video frame."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return transform(image).unsqueeze(0)  # Add batch dimension

def infer(model, image_tensor, device):
    """Run inference on a single image tensor."""
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

def main(args):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load model
    model = models.efficientnet_v2_s(weights=None, num_classes = 2)  # Change to desired EfficientNet variant
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Adjust for binary classification
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)

    # Open video file
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error opening video file {args.video_path}")
        return

    # Prepare video writer if output is specified
    if args.output_path:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(args.output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    classes = ['Defect', 'Normal']  # Adjust as per your dataset class names
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        image_tensor = load_image_from_frame(frame, transform)
        pred = infer(model, image_tensor, args.device)
        label = classes[pred]

        # Annotate the frame with the prediction
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display or save the frame
        if args.output_path:
            out.write(frame)
        else:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print(i,end="\r")
        i+=1

    # Release everything
    cap.release()
    if args.output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform video inference using an EfficientNet model.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights.')
    parser.add_argument('--output_path', type=str, help='Path to save the output video. If not specified, display the video.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for inference.')
    args = parser.parse_args()

    main(args)

#python effecientnet\inference.py --model_path effnet_1.pth --video_path ..\DataProcess\videos\defect\test\VID-20240720-WA0001.mp4 --output_path outputs\vid_1.avi
