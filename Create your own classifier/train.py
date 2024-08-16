import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

def get_data_loaders(data_directory, batch_size=32):
    """
    Loads the training and validation datasets from the specified directory,
    applies the necessary transformations, and returns the dataloaders.
    """
    # Define directory paths for training and validation data
    train_directory = os.path.join(data_directory, 'train')
    valid_directory = os.path.join(data_directory, 'valid')
    
    # Define the data augmentation and normalization transformations for training and validation datasets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    print("Loading datasets and applying transformations...")
    
    # Load datasets using ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_directory, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_directory, transform=data_transforms['valid']),
    }
    
    # Initialize dataloaders for batching and shuffling data
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=False),
    }
    
    print("Data loaders are ready.")
    return dataloaders, image_datasets

def initialize_pretrained_model(architecture='vgg16', hidden_units=512):
    """
    Loads a pretrained model from torchvision, freezes its parameters, and replaces
    the classifier with a custom classifier that matches the number of classes in the dataset.
    """
    print(f"Initializing a {architecture} model with {hidden_units} hidden units...")
    
    # Load a pretrained model based on the specified architecture
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_features = model.classifier[0].in_features
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_features = model.classifier.in_features
    else:
        raise ValueError("Model architecture not recognized. Choose 'vgg16' or 'densenet121'.")
    
    # Freeze the feature extraction layers so only the classifier is trained
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a new classifier to replace the pretrained model's classifier
    model.classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),  # 102 output classes for the flower dataset
        nn.LogSoftmax(dim=1)
    )
    
    print("Model initialization complete.")
    return model

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    """
    Trains the model using the specified dataloaders, criterion, optimizer, and device.
    Tracks training and validation loss and accuracy during each epoch.
    """
    print("Starting training...")
    model.to(device)  # Move the model to the appropriate device (CPU or GPU)
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0
        
        # Training loop
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Clear previous gradients
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()  # Accumulate loss
        
        # Validation loop after each epoch
        model.eval()  # Set model to evaluation mode
        valid_loss = 0
        accuracy = 0
        
        with torch.no_grad():  # Disable gradient calculation during validation
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item()
                
                # Calculate accuracy
                probabilities = torch.exp(outputs)
                top_p, top_class = probabilities.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch {epoch+1}/{num_epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
    
    print("Training complete.")

def save_checkpoint(model, dataset, architecture, hidden_units, save_directory):
    """
    Saves the trained model's checkpoint, including the architecture, hidden units,
    state_dict, and class_to_idx mapping.
    """
    checkpoint = {
        'architecture': architecture,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': dataset['train'].class_to_idx
    }
    
    save_path = os.path.join(save_directory, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to {save_path}")

def main():
    # Set up argument parsing for command-line inputs
    parser = argparse.ArgumentParser(description="Train an image classifier on a dataset.")
    parser.add_argument('data_dir', type=str, help='Path to the dataset folder')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the model checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture: vgg16 or densenet121')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Load data and create dataloaders
    dataloaders, dataset = get_data_loaders(args.data_dir)
    
    # Initialize the pretrained model
    model = initialize_pretrained_model(args.arch, args.hidden_units)
    
    # Set up the loss function (criterion) and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Determine the device (GPU or CPU)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    # Train the model
    train_model(model, dataloaders, criterion, optimizer, device, args.epochs)
    
    # Save the trained model checkpoint
    save_checkpoint(model, dataset, args.arch, args.hidden_units, args.save_dir)

if __name__ == '__main__':
    main()
