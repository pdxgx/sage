import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset, RandomSampler, Sampler
from sklearn.metrics import accuracy_score, DistanceMetric
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor, BallTree, KDTree, KernelDensity
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from collections import defaultdict
import inspect
import random
import numpy as np
import pandas as pd
import os
import re
import math
import scipy
import time
import warnings
import detectors
import timm
import pickle

class HamDataset(Dataset):
    '''
    adapted from: https://stackoverflow.com/questions/77528929/pytorch-imagefolder-vs-custom-dataset-from-single-folder
    '''
    def __init__(self, targ_dir, meta_file, transform=None):
        self.paths = sorted(list(Path(targ_dir).glob("*.jpg")))
        self.meta = pd.read_csv(meta_file)
        self.labels = list(map(self.get_label, self.paths))
        self.transform = transform
    
    def get_label(self, path):
        # make sure this function returns the label from the path
        match = re.search(r'ISIC_\d{7}', str(path)).group(0)
        label = self.meta.loc[self.meta['image_id'] == match]['label'].item() # matches image id to HAM10000 class label
        return label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        class_idx = self.labels[index]
        img_id = re.search(r'ISIC_\d{7}', str(path)).group(0) # matches 'image_id' column in metadata.csv

        if self.transform:
            return self.transform(img), class_idx, img_id
        else:
            return img, class_idx, img_id

class DdiDataset(Dataset):
    '''
    '''
    def __init__(self, targ_dir, meta_file, transform=None):
        self.paths = list(Path(targ_dir).glob("*.png"))
        self.meta = pd.read_csv(meta_file)
        self.labels = list(map(self.get_label, self.paths))
        self.transform = transform
    
    def get_label(self, path):
        # make sure this function returns the label from the path
        match = re.search(r'images/(.+)', str(path)).group(1) # image ID
        label = self.meta.loc[self.meta['DDI_file'] == match]['label'].item() 
        #label = int(label.iloc[0]) # pandas Series of len 1 to int
        return label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        class_idx = self.labels[index]
        img_id = re.search(r'images/(.+)', str(path)).group(1) # matches 'DDI_file' column in metadata.csv

        if self.transform:
            return self.transform(img), class_idx, img_id
        else:
            return img, class_idx, img_id

class HibaDataset(Dataset):
    '''
    '''
    def __init__(self, targ_dir, meta_file, transform=None):
        self.paths = list(Path(targ_dir).glob("*.jpg"))
        self.meta = pd.read_csv(meta_file)
        self.labels = list(map(self.get_label, self.paths))
        self.transform = transform
    
    def get_label(self, path):
        # make sure this function returns the label from the path
        match = re.search(r'ISIC_\d{7}', str(path)).group(0)
        label = self.meta.loc[self.meta['isic_id'] == match]['label'].item()
        #label = int(label.iloc[0]) # pandas Series of len 1 to int
        return label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        class_idx = self.labels[index]
        img_id = re.search(r'ISIC_\d{7}', str(path)).group(0) # matches 'isic_id' column in metadata.csv

        if self.transform:
            return self.transform(img), class_idx, img_id
        else:
            return img, class_idx, img_id

class UfesDataset(Dataset):
    '''
    '''
    def __init__(self, targ_dir, meta_file, transform=None):
        self.paths = list(Path(targ_dir).glob("*.png"))
        self.meta = pd.read_csv(meta_file)
        self.labels = list(map(self.get_label, self.paths))
        self.transform = transform
    
    def get_label(self, path):
        # make sure this function returns the label from the path
        match = re.search(r'images/(.+)', str(path)).group(1) # image ID
        label = self.meta.loc[self.meta['img_id'] == match]['label'].item()
        #label = int(label.iloc[0]) # pandas Series of len 1 to int
        return label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        class_idx = self.labels[index]
        img_id = re.search(r'images/(.+)', str(path)).group(1) # matches 'img_id' column in metadata.csv

        if self.transform:
            return self.transform(img), class_idx, img_id
        else:
            return img, class_idx, img_id

class OtherDataset(Dataset):
    '''
    '''
    def __init__(self, targ_dir, meta_file, transform=None):
        self.paths = list(Path(targ_dir).glob("*")) # gets all files in target dir
        self.meta = pd.read_csv(meta_file)
        self.labels = list(map(self.get_label, self.paths))
        self.transform = transform
    
    def get_label(self, path):
        # make sure this function returns the label from the path
        match = re.search(r'images/(.+)', str(path)).group(1) # image ID
        label = self.meta.loc[self.meta['img_id'] == match]['label'].item()
        return label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        class_idx = self.labels[index]
        img_id = re.search(r'images/(.+)', str(path)).group(1) # matches 'img_id' in metadata

        if self.transform:
            return self.transform(img), class_idx, img_id
        else:
            return img, class_idx, img_id

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        """
        Args:
            labels (list or array): List of labels corresponding to the dataset items.
            batch_size (int): Total batch size (must be divisible by number of classes).
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_classes = len(np.unique(self.labels))
        assert batch_size % self.num_classes == 0, "Batch size must be divisible by number of classes"
        self.samples_per_class = batch_size // self.num_classes
        
        # Map from label to indices
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)
        
        # Shuffle class indices initially
        for label in self.class_to_indices:
            random.shuffle(self.class_to_indices[label])

        self.num_batches = min(len(self.class_to_indices[c]) for c in self.class_to_indices) // self.samples_per_class

    def __iter__(self):
        # Pointers to where we are in each class list
        class_cursors = {label: 0 for label in self.class_to_indices}
        
        for _ in range(self.num_batches):
            batch = []
            for label in self.class_to_indices:
                start = class_cursors[label]
                end = start + self.samples_per_class
                batch.extend(self.class_to_indices[label][start:end])
                class_cursors[label] = end
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

class ResNetSAE(nn.Module):
    def __init__(self, latent_dim=32, num_classes=2, channels=3):
        super(ResNetSAE, self).__init__()
        self.latent_dim = latent_dim
        self.type = 'ResNetSAE'
        self.num_classes = num_classes
        
        # Load ResNet encoder
        resnet = models.resnet50(weights='IMAGENET1K_V1', progress=False)
        #resnet = models.resnet101(weights='IMAGENET1K_V1', progress=False)

        # Replace the fully connected layer
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, latent_dim)
        
        # Encoder is pre-trained ResNet
        self.encoder = resnet
        
        # Decoder
        self.decoder = nn.Sequential(
            # expand compressed embedding vector
            # First fc
            nn.Linear(latent_dim, 75),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Second fc
            nn.Linear(75, 150),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Third fc
            nn.Linear(150, 225),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Fourth fc
            nn.Linear(225, 300),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Fifth fc
            nn.Linear(300, 375),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Sixth fc
            nn.Linear(375, 512),
            nn.Unflatten(dim=1, unflattened_size=(32, 4, 4)),  # Output: [-1, 32, 4, 4]
            nn.LeakyReLU(),
            # First deconv: 4x4 -> 8x8
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Second deconv: 8x8 -> 16x16
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Third deconv: 16x16 -> 32x32
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Fourth deconv: 32x32 -> 64x64
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Fifth deconv: 64x64 -> 128x128
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Sixth deconv: 128x128 -> 256x256
            nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(),
            # Upsample layer: 256x256 -> 299x299
            nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1),  # Refinement convolution
        )

        if latent_dim < num_classes:
            # NN classifier with hourglass shape
            self.classifier = nn.Sequential(
                # First fc expands
                nn.Linear(self.latent_dim, 8),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Second fc expands
                nn.Linear(8, 16),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Third fc expands
                nn.Linear(16, 24),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Fourth fc expands
                nn.Linear(24, 32),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Fifth fc reduces
                nn.Linear(32, 24),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Sixth fc reduces
                nn.Linear(24, 16),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Seventh fc predicts
                nn.Linear(16, num_classes)
            )
        else:
            # NN classifier
            self.classifier = nn.Sequential(
                nn.Linear(self.latent_dim, 32),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 26),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(26, 20),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(20, 14),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(14, num_classes)
            )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logits = self.classifier(encoded)
        return encoded, decoded, logits

class InceptionSAE(nn.Module):
    def __init__(self, latent_dim=32, num_classes=2, channels=3):
        super(InceptionSAE, self).__init__()
        self.latent_dim = latent_dim
        self.type = 'InceptionSAE'
        self.num_classes = num_classes
        
        # Load ResNet encoder
        incept = models.inception_v3(weights='IMAGENET1K_V1', aux_logits=True) # causes weight loading error if no aux_logits

        # Replace the fully connected layer
        num_features = incept.fc.in_features
        incept.fc = nn.Linear(num_features, latent_dim)  # Map to latent_dim
        
        # Encoder is pre-trained Inception V3
        self.encoder = incept
        
        # Decoder
        self.decoder = nn.Sequential(
            # expand compressed embedding vector
            # First fc
            nn.Linear(latent_dim, 75),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Second fc
            nn.Linear(75, 150),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Third fc
            nn.Linear(150, 225),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Fourth fc
            nn.Linear(225, 300),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Fifth fc
            nn.Linear(300, 375),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Sixth fc
            nn.Linear(375, 512),
            nn.Unflatten(dim=1, unflattened_size=(32, 4, 4)),  # Output: [-1, 32, 4, 4]
            nn.LeakyReLU(),
            # First deconv: 4x4 -> 8x8
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Second deconv: 8x8 -> 16x16
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Third deconv: 16x16 -> 32x32
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Fourth deconv: 32x32 -> 64x64
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Fifth deconv: 64x64 -> 128x128
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Sixth deconv: 128x128 -> 256x256
            nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(),
            # Upsample layer: 256x256 -> 299x299
            nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1),  # Refinement convolution
        )

        if latent_dim < num_classes:
            # NN classifier with hourglass shape
            self.classifier = nn.Sequential(
                # First fc expands
                nn.Linear(self.latent_dim, 8),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Second fc expands
                nn.Linear(8, 16),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Third fc expands
                nn.Linear(16, 24),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Fourth fc expands
                nn.Linear(24, 32),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Fifth fc reduces
                nn.Linear(32, 24),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Sixth fc reduces
                nn.Linear(24, 16),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Seventh fc predicts
                nn.Linear(16, num_classes)
            )
        else:
            # NN classifier
            self.classifier = nn.Sequential(
                nn.Linear(self.latent_dim, 32),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 26),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(26, 20),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(20, 14),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(14, num_classes)
            )
    
    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded = self.decoder(encoded)
        logits = self.classifier(encoded)
        return encoded, decoded, logits

class VitSAE(nn.Module):
    def __init__(self, latent_dim=32, num_classes=2, channels=3):
        super(VitSAE, self).__init__()
        self.latent_dim = latent_dim
        self.type = 'VitSAE'
        self.num_classes = num_classes
        
        # Load ResNet encoder
        vit = models.vit_b_16(weights='IMAGENET1K_V1')

        # Replace the fully connected layer
        num_features = vit.heads.head.in_features
        vit.heads.head = nn.Linear(num_features, latent_dim)  # Map to latent_dim
        
        # Encoder is pre-trained ViT-Base with 16x16 input patch size
        self.encoder = vit
        
        # Decoder
        self.decoder = nn.Sequential(
            # expand compressed embedding vector
            # First fc
            nn.Linear(latent_dim, 75),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Second fc
            nn.Linear(75, 150),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Third fc
            nn.Linear(150, 225),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Fourth fc
            nn.Linear(225, 300),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Fifth fc
            nn.Linear(300, 375),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Sixth fc
            nn.Linear(375, 512),
            nn.Unflatten(dim=1, unflattened_size=(32, 4, 4)),  # Output: [-1, 32, 4, 4]
            nn.LeakyReLU(),
            # First deconv: 4x4 -> 8x8
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Second deconv: 8x8 -> 16x16
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Third deconv: 16x16 -> 32x32
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Fourth deconv: 32x32 -> 64x64
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Fifth deconv: 64x64 -> 128x128
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Sixth deconv: 128x128 -> 256x256
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Resizing layer: 256x256 -> 224x224
            nn.Conv2d(in_channels=32, out_channels=channels, kernel_size=3, padding=1), # drops channels from 32 -> 3
            nn.AdaptiveAvgPool2d((224, 224))
        )

        if latent_dim < num_classes:
            # NN classifier with hourglass shape
            self.classifier = nn.Sequential(
                # First fc expands
                nn.Linear(self.latent_dim, 8),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Second fc expands
                nn.Linear(8, 16),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Third fc expands
                nn.Linear(16, 24),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Fourth fc expands
                nn.Linear(24, 32),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Fifth fc reduces
                nn.Linear(32, 24),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Sixth fc reduces
                nn.Linear(24, 16),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                # Seventh fc predicts
                nn.Linear(16, num_classes)
            )
        else:
            # NN classifier
            self.classifier = nn.Sequential(
                nn.Linear(self.latent_dim, 32),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 26),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(26, 20),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(20, 14),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(14, num_classes)
            )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logits = self.classifier(encoded)
        return encoded, decoded, logits


class FcSAE(nn.Module):
    """
    Creates an autoencoder with a regressor component.
    """

    def __init__(self, latent_dim=2, in_features=None, num_classes=4):
        super(FcSAE, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.type = 'fcsae'

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(16, self.latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 16),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(64, in_features)
        )
        # NN classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, num_classes)
        )

    def forward(self, x):
        x = x.view(-1, self.in_features)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logit = self.classifier(encoded)
        return encoded, decoded, logit

class CenterLoss(nn.Module):
    def __init__(self, latent_dim, num_classes=10, metric='L2'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.metric = metric
        
        # Initialize the class centers randomly. Shape: (num_classes, feat_dim)
        self.centers = nn.Parameter(torch.randn(num_classes, latent_dim))
    
    def forward(self, embeddings, labels):
        """
        :param embeddings: The embeddings vectors outputted by the network (N x D)
        :param labels: The true labels corresponding to the input features (N)
        :return: The center loss (scalar)
        """
        # Get the centers of the classes for the given labels
        batch_size = embeddings.size(0)
        # Ensure labels are long and on the same device as centers
        labels = labels.long().to(self.centers.device)
        centers_batch = self.centers[labels]  # Shape: (batch_size, feat_dim)

        if self.metric == 'L1':
            # Compute the Manhattan distance
            center_loss = torch.sum(torch.abs(embeddings - centers_batch)) / batch_size
        elif self.metric == 'L2':
            # Compute the Euclidean distance
            center_loss = torch.sum((embeddings - centers_batch) ** 2) / batch_size
        else:
            raise TypeError('metric must be: "L1" or "L2"')
        
        return center_loss

    def update_centers(self, embeddings, labels, alpha=0.5):
        """
        Update the centers of the classes based on the current batch's features.
        :param embeddings: The embedding vectors outputted by the network (N x D)
        :param labels: The true labels corresponding to the input features (N)
        :param alpha: The update factor for the centers (learning rate)
        """
        batch_size = embeddings.size(0)
        embeddings = embeddings.to(self.centers.device)
        labels = labels.long().to(self.centers.device)
        # Ensure the centers don't require gradients
        with torch.no_grad():
            # Update class centers using the average of the current batch's features
            for i in range(batch_size):
                label = labels[i]
                # 1-alpha is momentum of previous center value
                self.centers.data[label] = (1 - alpha) * self.centers.data[label] + alpha * embeddings[i]

def train_sae_sequential(model, train_data, epochs=100, batch_size=64, sampler=None):
    '''
    '''
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # init loss fxns
    center_fn = CenterLoss(model.latent_dim).to(device)
    classifier_fn = nn.CrossEntropyLoss().to(device)
    decoder_fn = nn.MSELoss().to(device)
    
    lambda_center = 0.1 # Weight for center loss
    lambda_classifier = 0.5  # Weight for classification loss stage 1
    #lambda_classifier2 = 0.1  # Weight for classification loss stage 2
    lambda_decoder = 0.5  # Weight for reconstruction loss
    
    if sampler:
        train_loader = DataLoader(train_data, batch_sampler=sampler)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    history = list()
    start_time = time.time()

    print(f'Training {model.latent_dim}D model')

    ## STAGE 1
    print('\tStage 1 – Encoder and Classifier')
    
    # init optimizer for stage 1
    optimizer1 = optim.AdamW(
        list(model.encoder.parameters()) + 
        list(model.classifier.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )
    previous_centerloss = 1e4 # init center loss tracker with high value
    for epoch in range(epochs):
        print(f"\tEpoch {epoch + 1}/{epochs}:")
        model.train()
        
        # make running counters
        running_loss = 0
        running_classloss = 0
        running_centerloss = 0
        n_correct = 0
        total_samples = 0
        
        for data, labels, _ in train_loader:
            
            data, labels = data.to(device), labels.to(device) # push to device
            labels = labels.long() # ensures correct indexing
            encoded, decoded, logits = model(data) # forward pass
            
            # Calculate loss
            center_loss = center_fn(encoded, labels)
            class_loss = classifier_fn(logits, labels)
            loss = (center_loss * lambda_center) + (class_loss * lambda_classifier) # weighted loss terms
            
            # Backprop
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            
            # Update class centers after the batch
            center_fn.update_centers(encoded.detach(), labels)
            
            # Update total loss counters
            running_loss += loss.item() * data.size(0)
            running_classloss += class_loss.item() * data.size(0)
            running_centerloss += center_loss.item() * data.size(0)
            
            # Get accuracy
            _, preds = torch.max(logits, 1)
            n_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        # Get average batch loss
        total_loss = running_loss / total_samples
        total_classloss = running_classloss / total_samples
        total_centerloss = running_centerloss / total_samples
        accuracy = n_correct / total_samples * 100
        
        # print epoch results
        print(f"\t\tTotal Loss: {total_loss:.4f} | Class Loss: {total_classloss:.4f} | Center Loss: {total_centerloss:.4f} | Accuracy: {accuracy:.2f}%")
        history.append(total_loss)
        
        # Train stage 1 until center loss tracker stops improving
        if total_centerloss < previous_centerloss:
            previous_centerloss = total_centerloss # set new minimum
        elif total_centerloss > previous_centerloss:
            print(f'\t\tEarly stop for CenterLoss at epoch: {epoch + 1}')
            break # early stop
        
    ## STAGE 2
    print('\n\tStage 2 – Encoder, Decoder and Classifier')   
    
    # init stage 2 optimizer
    optimizer2 = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    for epoch in range(epochs):
        print(f"\tEpoch {epoch + 1}/{epochs}:")
        model.train()
        
        # make running counters
        running_loss = 0
        running_classloss = 0
        running_decoderloss = 0
        n_correct = 0
        total_samples = 0
        
        for data, labels, _ in train_loader:
            
            data, labels = data.to(device), labels.to(device) # push to device
            encoded, decoded, logits = model(data) # forward pass
            labels = labels.long()
            
            # Calculate loss
            class_loss = classifier_fn(logits, labels)
            decoder_loss = decoder_fn(decoded, data)
            loss = (decoder_loss * lambda_decoder) + (class_loss * lambda_classifier) # weighted loss terms
            
            # Backprop
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            
            # Update epoch counters
            running_loss += loss.item() * data.size(0)
            running_classloss += class_loss.item() * data.size(0)
            running_decoderloss += decoder_loss.item() * data.size(0)
            
            # Get accuracy
            _, preds = torch.max(logits, 1)
            n_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        # Get average batch loss
        total_loss = running_loss / total_samples
        total_classloss = running_classloss / total_samples
        total_decoderloss = running_decoderloss / total_samples
        accuracy = n_correct / total_samples * 100
        
        print(f"\t\tTotal Loss: {total_loss:.4f} | Class Loss: {total_classloss:.4f} | Recon Loss: {total_decoderloss:.4f} | Accuracy: {accuracy:.2f}%")
        history.append(total_loss)
    
    end_time = time.time()
    total_time = (end_time - start_time)/60
    print("Total train time: ", total_time, " mins")

    return model, history

def train_sae(model, train_data, val_data, epochs=50, lr=3e-4, recon_weight=0.5, task_weight=0.1):
    '''
    Basic one-stage training loop for supervised autoencoder. Combines weighted loss terms of 
    decoder (MSELoss) and classifier (CrossEntropyLoss) for backprop.
    '''
    # init variables
    task_fn = nn.CrossEntropyLoss()
    decoder_fn = nn.MSELoss()
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    
    # begin training
    history = []
    previous_valid = 1e4
    patience = 0
    for epoch in range(epochs):
        model.train()
        # make epoch stat counters
        total_loss = 0
        total_taskloss = 0
        total_reconloss = 0
        for data, labels in train_loader:
            encoded, decoded, logits = model(data)
            task_loss = task_fn(logits, labels) * task_weight
            recon_loss = decoder_fn(decoded, data) * recon_weight
            loss = task_loss + recon_loss
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update total loss counters
            total_loss += loss.item()
            total_taskloss += task_loss.item()
            total_reconloss += recon_loss.item()

        # average total loss across epoch batches
        total_loss /= len(train_loader)
        total_taskloss /= len(train_loader)
        total_reconloss /= len(train_loader)

        total_loss = round(float(total_loss), 5)
        total_taskloss = round(float(total_taskloss), 5)
        total_reconloss = round(float(total_reconloss), 5)

        # print epoch results
        print(f"\tEpoch {epoch+1}/{epochs}")
        print(f"\t\ttotal loss: {total_loss}, task loss: {total_taskloss}, recon loss: {total_reconloss}")
        
        history.append(total_loss)

        # make epoch stat counters
        total_valid_loss = 0
        for v_data, v_labels in val_loader:
            v_encoded, v_decoded, v_logits = model(v_data)
            valid_task_loss = task_fn(v_logits, v_labels) * task_weight
            valid_recon_loss = decoder_fn(v_decoded, v_data) * recon_weight
            valid_loss = valid_task_loss + valid_recon_loss
            # update total loss counters
            total_valid_loss += valid_loss.item()
        # average total loss across epoch batches
        total_valid_loss /= len(val_loader)
        total_valid_loss = round(float(total_valid_loss), 5)
        print(f"\t\tvalid loss: {total_valid_loss}, previous valid: {previous_valid}\n")
        if total_valid_loss > previous_valid:
            patience += 1
            previous_valid = total_valid_loss # reset previous validation loss tracker
            if patience > 1:
                print(f'Stopped early at epoch {epoch}')
                break
        else:
            patience = 0 # reset counter
            previous_valid = total_valid_loss # reset previous validation loss tracker
    end_time = time.time()
    total_time = (end_time - start_time)/60
    print("Total train time: ", total_time, " mins")

    return model, history

def train_sae_center(model, train_data, epochs=10, lr=3e-4, center_weight=0.1, recon_weight=100):
    '''
    Basic one-stage training loop for supervised autoencoder. Combines weighted loss terms of 
    encoder (CenterLoss), decoder (MSELoss) and classifier (CrossEntropyLoss) for backprop.
    '''
    # init variables
    center_fn = CenterLoss(model.latent_dim, num_classes=model.num_classes, metric='L1')
    task_fn = nn.CrossEntropyLoss()
    decoder_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    # begin training
    start_time = time.time()
    print(f'Training {model.latent_dim}D model')
    print('\tTraining Embedding, Decoder and Classifier')
    history = []
    for epoch in range(epochs):
        model.train()
        # make epoch stat counters
        total_loss = 0
        total_centerloss = 0
        total_taskloss = 0
        total_reconloss = 0
        for data, labels in train_loader:
            encoded, decoded, logits = model(data)
            center_loss = center_fn(encoded, labels) * center_weight # upweight center loss
            task_loss = task_fn(logits, labels)
            recon_loss = decoder_fn(decoded, data) * recon_weight  # upweight reconstruction loss
            loss = center_loss + task_loss + recon_loss

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update class centers after the batch
            center_fn.update_centers(encoded.detach(), labels)
            
            # update total loss counters
            total_loss += loss.item()
            total_centerloss += center_loss.item()
            total_taskloss += task_loss.item()
            total_reconloss += recon_loss.item()

        # average total loss across epoch batches
        total_loss /= len(train_loader)
        total_centerloss /= len(train_loader)
        total_taskloss /= len(train_loader)
        total_reconloss /= len(train_loader)

        total_loss = round(float(total_loss), 5)
        total_centerloss = round(float(total_centerloss), 5)
        total_taskloss = round(float(total_taskloss), 5)
        total_reconloss = round(float(total_reconloss), 5)

        # print epoch results
        print(f"\tEpoch {epoch+1}/{epochs}")
        print(f"\t\ttotal loss: {total_loss}\n\t\tcenter loss: {total_centerloss}, task loss: {total_taskloss}, recon loss: {total_reconloss}")
        
        history.append(total_loss)
    
    end_time = time.time()
    total_time = (end_time - start_time)/60
    print("Total train time: ", total_time, " mins")

    return model, history

# Denormalization helper (ImageNet stats)
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

# Visualization function
def show_reconstruction(original, reconstructed, n=5):
    """
    Displays n pairs of (original, reconstructed) images.
    
    Args:
        original (Tensor): Batch of normalized input images [B, 3, H, W]
        reconstructed (Tensor): Batch of reconstructed images [B, 3, H, W]
        n (int): Number of images to show
    """
    original = denormalize(original.detach().cpu())
    reconstructed = denormalize(reconstructed.detach().cpu())
    
    original = torch.clamp(original, 0, 1)
    reconstructed = torch.clamp(reconstructed, 0, 1)

    plt.figure(figsize=(n * 3, 6))

    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        img = original[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title("Original")
        ax.axis("off")

        # Reconstructed
        ax = plt.subplot(2, n, n + i + 1)
        img = reconstructed[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title("Reconstructed")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
