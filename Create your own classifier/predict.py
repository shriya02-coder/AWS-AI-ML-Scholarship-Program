import argparse
import torch
from torchvision import models
from torch import nn
from PIL import Image
from torchvision import transforms
import json
import os

def load_checkpoint(checkpoint_path):
    """
    Loads a model checkpoint from the specified path and rebuilds the model with
    the architecture and classifier configuration stored in the checkpoint.
    """
    print("Loading model checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    
    # Load the pre-trained model architecture
    model = models.__dict__[checkpoint['architecture']](pretrained=True)
    
    # Rebuild the classifier based on the architecture
    if checkpoint['architecture'] == 'vgg16':
        input_features = model.classifier[0].in_features
    elif checkpoint['architecture'] == 'densenet121':
        input_features = model.classifier.in_features
    else:
        raise ValueError("Unsupported architecture in the checkpoint.")
    
    model.classifier = nn.Sequential(
        nn.Linear(input_features, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(checkpoint['hidden_units'], 102),  # 102 output classes
        nn.LogSoftmax(dim=1)
    )
    
    # Load the model state from the checkpoint
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    print("Model loaded successfully.")
    return model

def process_image(image_path):
    """
    Scales, crops, and normalizes an image for use in a PyTorch model.
    Returns the processed image tensor.
    """
    print(f"Processing image: {image_path}")
    
    # Load the image and apply transformations
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return preprocess(image)

def predict(image_path, model, device, top_k=1):
    """
    Predicts the top K classes for the given image using the trained model.
    Returns the probabilities and corresponding classes.
    """
    print("Running inference...")
    
    # Preprocess the image
    image = process_image(image_path).unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Move the image tensor to the correct device
    
    # Move the model to the correct device
    model.to(device)
    
    # Set the model to evaluation mode and disable gradients
    model.eval()
    with torch.no_grad():
        output = model(image)
    
    # Convert model output to probabilities
    probabilities = torch.exp(output)
    
    # Get the top K probabilities and their corresponding class indices
    top_prob, top_classes = probabilities.topk(top_k, dim=1)
    
    # Convert class indices to actual class labels using the class_to_idx mapping
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[cls.item()] for cls in top_classes[0]]
    
    print("Inference complete.")
    return top_prob.squeeze().tolist(), top_classes

def load_category_names(category_names_path):
    """
    Loads a JSON file that maps category indices to real names (e.g., flower names).
    Returns a dictionary mapping class indices to names.
    """
    with open(category_names_path, 'r') as f:
        category_names = json.load(f, strict=False)  # Load with strict=False to avoid errors
    print("Category names loaded successfully.")
    return category_names

def main():
    # Set up argument parsing for command-line inputs
    parser = argparse.ArgumentParser(description="Predict the class of an input image using a trained model.")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Determine the device (GPU or CPU)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    # Load the model checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Make predictions on the input image
    top_probabilities, top_classes = predict(args.image_path, model, device, args.top_k)
    
    # Load the category names if provided, otherwise use class indices
    if args.category_names:
        category_names = load_category_names(args.category_names)
        top_class_names = [category_names[str(cls)] for cls in top_classes]
    else:
        top_class_names = top_classes  # Default to class indices
    
    # Print the most likely class and the top K classes
    print(f"\nMost likely class: {top_class_names[0]} with probability {top_probabilities[0]:.4f}")
    print("\nTop K classes:")
    for i in range(len(top_probabilities)):
        print(f"{i + 1}. {top_class_names[i]}: {top_probabilities[i]:.4f}")

if __name__ == '__main__':
    main()
