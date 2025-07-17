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
        help="Path to directory with HAM10000 dataset images."
    )
    parser.add_argument(
        "--metafile",
        help="Path to comma-separated file with HAM10000 image labels."
    )
    parser.add_argument(
        "--savedir",
        help="Path to folder for saving trained HAM10000 SAGE model and train history."
    )
    parser.add_argument(
        "--encoder",
        help="Type of encoder to use for SAGE model (one of ['ResNet', 'Inception', 'ViT'])."
    )
    parser.add_argument(
        "--dim",
        default=32,
        help="Dimensions of latent space (encoder output). Default is 32."
    )
    args = parser.parse_args()

    assert args.encoder in ['ResNet', 'Inception', 'ViT']
    
    # Transform used to train DeepDerm model (based on Inception V3) in DDI paper, missing cutout of upright rectangle
    paper_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(0, 359)),    
        transforms.RandomVerticalFlip(p=0.5),       
        transforms.Resize(299),                 
        transforms.CenterCrop(299),            
        transforms.ToTensor(),                  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if args.encoder == 'ViT':
        # ViT needs input size of 224x224
        transform = transforms.Compose([
            transforms.Resize(224),               
            transforms.CenterCrop(224),            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # use ImageNet values
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(299),               
            transforms.CenterCrop(299),            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # use ImageNet values
        ])
    
    # Split dataset into train (90% train) and test
    ham_dataset = HamDataset(args.imagedir, args.metafile, transform=transform)
    train_indices, test_indices = train_test_split(
        np.arange(len(ham_dataset)),
        test_size=0.1,
        stratify=ham_dataset.labels,
        random_state=33 # makes split reproducible
    )
    
    # Create subsets for train and test
    train_dataset = Subset(ham_dataset, train_indices)
    test_dataset = Subset(ham_dataset, test_indices)
    
    # Init balanced batch sampler
    batch_size = 63 # must be divisible by n classes
    train_labels = [ham_dataset.labels[i] for i in train_indices] # returns labels for train images
    balanced_sampler = BalancedBatchSampler(train_labels, batch_size=batch_size)

    # Init model
    if args.encoder == 'ResNet':
        model = ResNetSAE(latent_dim=args.dim, num_classes=7, channels=3)
    elif args.encoder == 'Inception':
        model = InceptionSAE(latent_dim=args.dim, num_classes=7, channels=3)
    elif args.encoder == 'ViT':
        model = VitSAE(latent_dim=args.dim, num_classes=7, channels=3)
    
    # Run 2-step training process
    model, history = train_sae_sequential(model, train_dataset, epochs=100, batch_size=batch_size, sampler=balanced_sampler)
    
    # Save trained model as state dict
    model_path = os.path.join(args.savedir, f'{model.type}_{args.dim}D')
    torch.save(model.state_dict(), model_path)
    
    # Save history as pickled object
    hist_path = os.path.join(args.savedir, f'{model.type}_{args.dim}D_history.pkl')
    with open(hist_path, 'wb') as file:
        pickle.dump(history, file)

if __name__ == '__main__':
    main()
