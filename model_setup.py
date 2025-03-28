import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset, RandomSampler, Sampler
from sklearn.metrics import accuracy_score, DistanceMetric
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor, BallTree, KDTree, KernelDensity
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from temperature_scaling import *
from tqdm import tqdm
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import inspect
import random
import numpy as np
import pandas as pd
import os
import math
import scipy
import time
import warnings
import detectors
import timm
import pickle

class CustomTensorDataset():
    """
    TensorDataset with application of transforms.
    """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        # Handle batch indexing
        if isinstance(index, (list, np.ndarray, torch.Tensor)):
            index = np.array(index)
            x = self.data[index]
            y = self.targets[index]
            # Apply transformations batch-wise
            if self.transform:
                x = torch.stack([self.transform(sample) for sample in x])
            return x, y

        # Handle single indexing
        x = self.data[index]
        y = self.targets[index]
        # Ensure image has a channel dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Reshape to (1, H, W)
        # set seeds
        random.seed(33)
        torch.manual_seed(33)
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        if isinstance(self.data, np.ndarray):
            return self.data.shape[0]  # Use .shape for NumPy arrays
        elif isinstance(self.data, torch.Tensor):
            return self.data.size(0)  # Use .size(0) for PyTorch tensors
        else:
            raise TypeError("Unsupported data type for 'data'. Must be np.ndarray or torch.Tensor.")

class ConvSAE(nn.Module):
    def __init__(self, latent_dim=2, num_classes=10, channels=1, width=28):
        super(ConvSAE, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.in_features = (width ** 2) * channels
        self.type = 'convsae'
        self.num_classes = num_classes
        
        ## Encoder
        self.encoder = nn.Sequential(
            # two convolutional layers
            nn.Conv2d(in_channels=channels, out_channels=width, kernel_size=3, stride=1, padding=1),  # Output: [w, w, w]
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1),  # Output: [w, w, w]
            nn.BatchNorm2d(width), # Output: [w, w, w],
            nn.LeakyReLU()
        )
        # maxpool
        self.maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True) # Output: [w, w/2, w/2], indices
        self.maxout = int((width ** 3) / 4)
        # dense layers
        self.fc_encode = nn.Sequential(
            nn.Flatten(), # Output: [w**3/4]
            nn.Linear(self.maxout, 196),
            nn.LeakyReLU(),
            nn.Linear(196, latent_dim)
        )
        
        ## Decoder
        self.fc_decode = nn.Sequential(
            # dense layers
            nn.Linear(latent_dim, 196),
            nn.LeakyReLU(),
            nn.Linear(196, self.maxout), # Output: [w**3/4]
            # reshape data
            nn.Unflatten(dim=1, unflattened_size = (width, int(width/2), int(width/2))) # Output: [w, w/2, w/2]
        )
        self.maxunpool = nn.MaxUnpool2d(kernel_size=2) # Output: [w, w, w]
        # two de-convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(in_channels=width, out_channels=channels, kernel_size=3, stride=1, padding=1),
        )
        
        ## Two-layer MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 20),
            nn.LeakyReLU(),
            nn.Linear(20, num_classes)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x, indices = self.maxpool(x)
        encoded = self.fc_encode(x)
        logits = self.classifier(encoded)
        decoded = self.fc_decode(encoded)
        decoded = self.maxunpool(decoded, indices)
        decoded = self.decoder(decoded)
        return encoded, decoded, logits

class SmallConvSAE(nn.Module):
    def __init__(self, latent_dim=2, num_classes=10, channels=1, width=28):
        super(ConvSAE, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.type = 'smallconvsae'
        self.num_classes = num_classes
        
        ## Encoder
        self.encoder = nn.Sequential(
            # three convolutional layers
            nn.Conv2d(in_channels=channels, out_channels=width, kernel_size=3, padding=1, stride=2),  # Output: [-1, w, 14, 14]
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=width, out_channels=2*width, kernel_size=3, padding=1, stride=2), # Output: [-1, w*2, 7, 7]
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=2*width, out_channels=2*width, kernel_size=3, padding=1, stride=2), # Output: [-1, w*2, 4, 4]
            nn.LeakyReLU(),
            nn.Flatten(), # Image grid to single feature vector, Output: [-1, 2*16*width]
            nn.Linear(2*16*width, latent_dim) # Output: [-1, latent_dim]
        )

        ## Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2*16*width),
            nn.Unflatten(dim=1, unflattened_size = (2*width, 4, 4)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=2*width, out_channels=width, kernel_size=3, padding=1, stride=2), # Output: [-1, w, 7, 7]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=width, out_channels=width, kernel_size=4, padding=1, stride=2), # Output: [-1, w, 14, 14]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=width, out_channels=channels, kernel_size=4, padding=1, stride=2), # Output: [-1, channels, w, w]
        )

        ## Two-layer MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 20),
            nn.LeakyReLU(),
            nn.Linear(20, num_classes)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logits = self.classifier(encoded)
        return encoded, decoded, logits

class ResNetConvSAE(nn.Module):
    def __init__(self, latent_dim=2, num_classes=10, channels=3, width=32):
        super(ResNetConvSAE, self).__init__()
        self.latent_dim = latent_dim
        self.type = 'resnetconvsae'
        self.num_classes = num_classes
        
        # Load ResNet encoder
        resnet = models.resnet18(weights='IMAGENET1K_V1', progress=False)
        #resnet = models.resnet18(progress=False)

        # Replace the fully connected layer
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_features, latent_dim),  # Map to latent_dim
        )
        # Encoder is ResNet
        self.encoder = resnet
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2*16*width),
            nn.Unflatten(dim=1, unflattened_size = (64, 4, 4)), # Output: [-1, w*2, w/8, w/8]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=width*2, out_channels=width, kernel_size=4, padding=1, stride=2), # Output: [-1, w, w/4, w/4]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=width, out_channels=width, kernel_size=4, padding=1, stride=2), # Output: [-1, w, w/2, w/2]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=width, out_channels=channels, kernel_size=4, padding=1, stride=2), # Output: [-1, channels, w, w]
        )
        
        ## Two-layer MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 20),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(20, num_classes)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logits = self.classifier(encoded)
        return encoded, decoded, logits

class RegressorSAE(nn.Module):
    """
    Creates an autoencoder with a regressor component.
    """
    def __init__(self, latent_dim=2, in_features=8):
        super(RegressorSAE, self).__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.type='regressor'
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, in_features)
        )
        # NN regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        x = x.view(-1, self.in_features)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logit = self.regressor(encoded)
        return encoded, decoded, logit


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
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
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

class Torch_ResNet(nn.Module):
    """
    adapted from:
        https://colab.research.google.com/github/kjamithash/Pytorch_DeepLearning_Experiments/blob/master/
        FashionMNIST_ResNet_TransferLearning.ipynb#scrollTo=LzkK82Swc4ca
    """
    def __init__(self, pretrained=False, in_channels=1, num_classes=10, layers=18):
        super(Torch_ResNet, self).__init__()
        self.num_classes = num_classes
        self.type = 'resnet'
        
        # Load a pretrained resnet model from torchvision.models in Pytorch
        if pretrained == True:
            if layers == 18:
                self.model = models.resnet18(weights='DEFAULT')
            elif layers == 34:
                self.model = models.resnet34(weights='DEFAULT')
            # Freeze earlier layers
            for param in self.model.parameters():
                param.requires_grad = False  # Freeze all layers
            # Unfreeze the final layer
            for param in self.model.fc.parameters():
                param.requires_grad = True
        else:
            if layers == 18:
                self.model = models.resnet18()
            elif layers == 34:
                self.model = models.resnet34()
        
        # original first layer: self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Change the output layer to output c classes instead of 1000 classes
        in_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(in_ftrs, num_classes)
        self.model.maxpool = nn.Identity()  # Skip max-pooling for small images

    def forward(self, x):
        return self.model(x)

# for checking the dimensions of intermediate outputs
def get_activation(name):
    '''
    Usage:
        print(model.encoder[-1])
        activation = {}
        model.encoder[-1].register_forward_hook(get_activation('out'))
        for img, label in dl:
            output = model(img)
            print(activation['out'].shape)
            break
    '''
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def hard_triplets(encoded, y):
    distance_matrix = torch.cdist(encoded, encoded, p=1) # Create square distance matrix
    anchors = []
    positives = []
    negatives = []
    for j in range(len(y)):
        anchor_label = y[j].item()
        anchor_distance = distance_matrix[j] # distance between anchor and all other points
        # Hardest positive (farthest in the same class)
        hardest_positive_idx = (y == anchor_label).nonzero(as_tuple=True)[0] # all same class indices
        hardest_positive_idx = hardest_positive_idx[hardest_positive_idx != j] # exclude anchor point index
        if hardest_positive_idx.numel() == 0:
            continue  # Skip if no positives in batch
        hardest_positive = hardest_positive_idx[anchor_distance[hardest_positive_idx].argmax()]
        # Hardest negative (closest from different class)
        hardest_negative_idx = (y != anchor_label).nonzero(as_tuple=True)[0] # all diff class indices
        if hardest_negative_idx.numel() == 0:
            continue  # Skip if no negatives in batch
        hardest_negative = hardest_negative_idx[anchor_distance[hardest_negative_idx].argmin()] # index of closest different class
        # load selected
        anchors.append(encoded[j])
        positives.append(encoded[hardest_positive])
        negatives.append(encoded[hardest_negative])
    
    # Convert lists to tensors
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    
    return anchors, positives, negatives

def semihard_triplets(encoded, y, margin=1.0):
    distance_matrix = torch.cdist(encoded, encoded, p=1)  # Create square distance matrix
    anchors = []
    positives = []
    negatives = []
    for j in range(len(y)):
        anchor_label = y[j].item()
        anchor_distance = distance_matrix[j]  # distance between anchor and all other points
        
        # Hardest positive (farthest in the same class)
        positive_indices = (y == anchor_label).nonzero(as_tuple=True)[0]  # all same class indices
        positive_indices = positive_indices[positive_indices != j]  # exclude anchor point index
        if positive_indices.numel() == 0:
            continue  # Skip if no positives in batch
        hardest_positive = positive_indices[anchor_distance[positive_indices].argmax()]
        hardest_positive_distance = anchor_distance[hardest_positive]

        # Semi-hard negative (between the hardest positive and hardest positive + margin)
        negative_indices = (y != anchor_label).nonzero(as_tuple=True)[0]  # all diff class indices
        if negative_indices.numel() == 0:
            continue  # Skip if no negatives in batch
        negative_distances = anchor_distance[negative_indices]
        
        # Select negatives that are closer than the hardest positive and violate the margin
        semihard_negative_mask = negative_distances < (hardest_positive_distance + margin)
        semihard_negative_mask &= negative_distances > hardest_positive_distance
        semihard_negative_indices = negative_indices[semihard_negative_mask]
        
        if semihard_negative_indices.numel() == 0:
            continue  # Skip if no semi-hard negatives found
        #semihard_negative = semihard_negative_indices[negative_distances[semihard_negative_mask].argmin()] # hardest semihard negative
        
        # randomly select a semihard negative
        semihard_negative = np.random.choice(semihard_negative_indices, 1)[0]
        
        # Append triplet components
        anchors.append(encoded[j])
        positives.append(encoded[hardest_positive])
        negatives.append(encoded[semihard_negative])

    # Convert lists to tensors
    if len(anchors) == 0:  # Handle edge case where no triplets are found
        return None, None, None
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    return anchors, positives, negatives

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
        # Ensure the centers don't require gradients
        with torch.no_grad():
            # Update class centers using the average of the current batch's features
            for i in range(batch_size):
                label = labels[i]
                # 1-alpha is momentum of previous center value
                self.centers.data[label] = (1 - alpha) * self.centers.data[label] + alpha * embeddings[i]

def train_sae_sequential(model, train_data, epochs=10):
    '''
    '''
    batches = 64
    center_fn = CenterLoss(model.latent_dim)
    task_fn = nn.CrossEntropyLoss()
    decoder_fn = nn.MSELoss()
    train_loader = DataLoader(train_data, batch_size=batches, shuffle=True)
    history = dict()
    start_time = time.time()

    print(f'Training {model.latent_dim}D model')
    print('\tTraining Encoder and Classifier (Classification + Contrastive Loss)')
    history1 = []
    # init optimizer for stage 1
    optimizer1 = optim.Adam(
        list(model.encoder.parameters()) + 
        list(model.classifier.parameters()),
        lr=3e-4
    )
    for epoch in range(epochs):
        model.train()
        # make epoch stat counters
        total_loss = 0
        total_taskloss = 0
        total_centerloss = 0
        for data, labels in train_loader:
            encoded, _, logits = model(data)
            center_loss = center_fn(encoded, labels)
            center_loss *= 0.1 # downweight center loss contribution to gradient
            task_loss = task_fn(logits, labels)
            loss = center_loss + task_loss
            # backprop
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            # Update class centers after the batch
            center_fn.update_centers(encoded.detach(), labels)
            # update total loss counters
            total_loss += loss.item()
            total_taskloss += task_loss.item()
            total_centerloss += center_loss.item()
        
        # average total loss across epoch batches
        total_loss /= len(train_loader)
        total_taskloss /= len(train_loader)
        total_centerloss /= len(train_loader)
        # print epoch results
        print(f"\tEpoch {epoch+1}/{epochs}")
        print(f"\t\ttotal loss: {total_loss}, task loss: {total_taskloss}, center loss: {total_centerloss},")
        history1.append(round(float(total_loss), 4))
    history['stage1'] = history1
        
    print('\n\tTraining Encoder and Decoder (Classification + Reconstruction Loss)')
    # freeze classifier weights
    #for param in model.classifier.parameters():
        #param.requires_grad = False
        
    # init stage 2 optimizer
    optimizer2 = optim.Adam(
        [
            {'params': model.encoder.parameters(), 'lr': 1e-4},  # Low learning rate for encoder
            {'params': model.classifier.parameters(), 'lr': 5e-5},  # Lowest learning rate for classifier
            {'params': model.decoder.parameters(), 'lr': 3e-4}
        ]
    )
    history2 = []
    for epoch in range(epochs):
        model.train()
        # make epoch stat counters
        total_loss = 0
        total_taskloss = 0
        total_reconloss = 0
        for data, labels in train_loader:
            _, decoded, logits = model(data)
            task_loss = task_fn(logits, labels)
            decoder_loss = decoder_fn(decoded, data)
            loss = task_loss + decoder_loss
            # backprop loss
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            # update epoch counters
            total_loss += loss.item()
            total_taskloss += task_loss.item()
            total_reconloss += decoder_loss.item()
        
        total_loss /= len(train_loader) # get per-batch loss
        total_taskloss /= len(train_loader)
        total_reconloss /= len(train_loader)
        
        print(f"\tEpoch {epoch+1}/{epochs}")
        print(f"\t\ttotal loss: {total_loss}, task loss: {total_taskloss}, decoder loss: {total_reconloss},")
        history2.append(round(float(total_loss), 4))
    history['stage2'] = history2
    
    end_time = time.time()
    total_time = (end_time - start_time)/60
    print("Total train time: ", total_time, " mins")

    return model, history

def train_sae_regressor(model, train_data, epochs=50):
    '''
    '''
    batches = 64
    task_fn = nn.MSELoss()
    decoder_fn = nn.MSELoss()
    train_loader = DataLoader(train_data, batch_size=batches, shuffle=True)
    history = []
    start_time = time.time()

    print(f'Training {model.latent_dim}D model')
    print('\tTraining Encoder, Decoder and Regressor')
    history = []
    # init optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(epochs):
        model.train()
        # make epoch stat counters
        total_loss = 0
        total_taskloss = 0
        total_reconloss = 0
        for data, labels in train_loader:
            encoded, decoded, logits = model(data)
            logits = logits.view([data.size(0)]) # reshape logits before loss calculation
            task_loss = task_fn(logits, labels)
            recon_loss = decoder_fn(decoded, data)
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

        # print epoch results
        print(f"\tEpoch {epoch+1}/{epochs}")
        print(f"\t\ttotal loss: {total_loss}, task loss: {total_taskloss}, recon loss: {total_reconloss}")
        
        history.append(round(float(total_loss), 4))
    
    end_time = time.time()
    total_time = (end_time - start_time)/60
    print("Total train time: ", total_time, " mins")

    return model, history

def train_sae(model, train_data, epochs=10):
    '''
    '''
    # init variables
    task_fn = nn.CrossEntropyLoss()
    decoder_fn = nn.MSELoss()
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    start_time = time.time()
    # begin training
    print(f'Training {model.latent_dim}D model')
    print('\tTraining Encoder, Decoder and Classifier')
    history = []
    for epoch in range(epochs):
        model.train()
        # make epoch stat counters
        total_loss = 0
        total_taskloss = 0
        total_reconloss = 0
        for data, labels in train_loader:
            encoded, decoded, logits = model(data)
            task_loss = task_fn(logits, labels)
            recon_loss = decoder_fn(decoded, data) * 100.0  # upweight reconstruction loss
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

        # print epoch results
        print(f"\tEpoch {epoch+1}/{epochs}")
        print(f"\t\ttotal loss: {total_loss}, task loss: {total_taskloss}, recon loss: {total_reconloss}")
        
        history.append(round(float(total_loss), 4))
    
    end_time = time.time()
    total_time = (end_time - start_time)/60
    print("Total train time: ", total_time, " mins")

    return model, history

def get_resnet_output(resnet, datasets, dataset_names, return_probs=True, save=False, save_path=None):
    # get ResNet predictions
    name_arr, preds_arr, probs_arr, conf_arr, lbls_arr = [], [], [], [], []
    sm = nn.Softmax(dim=1) # row-wise softmax
    for name in dataset_names:
        dloader = DataLoader(datasets[name], batch_size=64, shuffle=False)
        name_arr.append([name]*len(datasets[name]))
        resnet.eval()
        predictions, labels = [], []
        for imgs, lbls in dloader:
            with torch.no_grad():
                probs = resnet.forward(imgs)
                probs = sm(probs) # apply softmax to output
                predictions.append(probs)
                labels.extend(lbls)
        # collapse tensors
        all_probs = torch.cat(predictions).numpy()
        probs_arr.append(all_probs) # holds confidence for all classes
        conf_arr.append(np.max(all_probs, axis=1)) # keeps only highest confidence value
        preds_arr.append(all_probs.argmax(axis=1)) # assigns prediction class
        lbls_arr.append(np.array(labels))
    
    if return_probs == True: # export predicted probabilities for all classes
        col_names = ['data', 'preds', 'labels'] + [f'conf{i}' for i in range(10)]
        resnet_df = pd.DataFrame(columns=col_names)
        resnet_df['data'] = np.concatenate(name_arr, axis=None)
        resnet_df['preds'] = np.concatenate(preds_arr, axis=None)
        resnet_df['labels'] = np.concatenate(lbls_arr, axis=None)
        # add predicted probs
        probs_vstack = stack = np.vstack(probs_arr)
        for i in range(10): # iterate through classes
            iprobs = np.concatenate(probs_vstack[:,i], axis=None)
            resnet_df[f'conf{i}'] = iprobs
    else: # retain only max pred prob value
        resnet_df = pd.DataFrame(
            {
                'data':np.concatenate(name_arr, axis=None),
                'preds':np.concatenate(preds_arr, axis=None),
                'labels':np.concatenate(lbls_arr, axis=None),
                'conf':np.concatenate(conf_arr, axis=None)
            }
        )
    # column for binary True/False if prediction is correct
    resnet_df['resnet_correct'] = resnet_df['preds'] == resnet_df['labels']
        
    if save==True:
        # pickle and save
        with open(save_path, 'wb') as df_stream:
            pickle.dump(resnet_df, df_stream, pickle.HIGHEST_PROTOCOL) # use highest protocol
    
    return resnet_df