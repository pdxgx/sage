from model_setup import *
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
import torchvision
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagedir",
        help="Directory with HAM10000 dataset images."
    )
    parser.add_argument(
        "--metafile",
        help="File with binary image labels (malignant [1] vs. benign [0])."
    )
    parser.add_argument(
        "--modelpth",
        help="Path to save trained HAM10000 SAGE model state dict."
    )
    args = parser.parse_args()
    
    # Define the transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),    
        transforms.RandomVerticalFlip(),        
        transforms.Resize(320),                 
        transforms.CenterCrop(299),            
        transforms.ToTensor(),                  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (ImageNet stats, same as DDI paper)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(320),               
        transforms.CenterCrop(299),            
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Split dataset into train and test (80% train, 20% test)
    all_dataset = DermoDataset(args.imagedir, args.metafile, transform=None)
    train_indices, test_indices = train_test_split(
        np.arange(len(all_dataset)),
        test_size=0.2,
        stratify=all_dataset.labels,
        random_state=33
    )
    
    # Create subsets for train and test
    train_dataset = Subset(all_dataset, train_indices)
    test_dataset = Subset(all_dataset, test_indices)
    
    # Dynamically apply transforms
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform
    
    # Init balanced batch sampler
    batch_size = 32
    train_labels = [all_dataset.labels[i] for i in train_indices] # returns labels for train images
    balanced_sampler = BalancedBatchSampler(train_labels, batch_size=batch_size)

    # Create train DataLoader using balanced sampler
    train_loader = DataLoader(train_dataset, batch_sampler=balanced_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Init parameters
    lr = 1e-4
    weight_decay = 1e-5
    num_epochs = 20
    lambda_classifier = 1.0  # Weight for classification loss
    lambda_decoder = 1.0  # Weight for reconstruction loss
    dims = 20
    
    # Model
    model = ResNetSAE(latent_dim=dims, num_classes=2, channels=3, width=299)#.cuda()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Loss functions
    classifier_fn = nn.CrossEntropyLoss()
    decoder_fn = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        running_class_loss = 0.0
        running_recon_loss = 0.0
        running_total_loss = 0.0
        n_correct = 0
        total_samples = 0
    
        for images, labels in train_loader:
            
            #images, labels = images.cuda(), labels.cuda()
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            encoded, decoded, logits = model(images)
            # Get loss
            class_loss = classifier_fn(logits, labels)
            recon_loss = decoder_fn(decoded, images)
            total_loss = lambda_classifier * class_loss + lambda_decoder * recon_loss
            # Backward pass
            total_loss.backward()
            optimizer.step()
            # Track losses and accuracy
            running_class_loss += class_loss.item() * images.size(0)
            running_recon_loss += recon_loss.item() * images.size(0)
            running_total_loss += total_loss.item() * images.size(0)
            # Get accuracy
            _, preds = torch.max(logits, 1)
            n_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
        # Average loss and accuracy
        class_loss = running_class_loss / total_samples
        recon_loss = running_recon_loss / total_samples
        total_loss = running_total_loss / total_samples
        accuracy = n_correct / total_samples * 100
        
        print(f"Train Loss: {total_loss:.4f} | Class Loss: {class_loss:.4f} | Recon Loss: {recon_loss:.4f} | Accuracy: {accuracy:.2f}%")
    
    # Save trained model as state dict
    torch.save(model.state_dict(), args.modelpth)

if __name__ == '__main__':
    main()