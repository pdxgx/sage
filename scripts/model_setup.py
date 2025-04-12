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
from temperature_scaling import *
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

class DermoDataset(Dataset):
    '''
    adapted from: https://stackoverflow.com/questions/77528929/pytorch-imagefolder-vs-custom-dataset-from-single-folder
    '''
    def __init__(self, targ_dir, meta_file, transform=None):
        self.paths = list(Path(targ_dir).glob("*.jpg"))
        self.meta = pd.read_csv(meta_file)
        self.labels = list(map(self.get_label, self.paths))
        self.transform = transform
    
    def get_label(self, path):
        # make sure this function returns the label from the path
        match = re.search(r'ISIC_\d{7}', str(path))
        label = self.meta.loc[self.meta['image_id'] == match.group()]['label'].item() # matches image id to binary label
        return label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        class_idx = self.labels[index]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

class DdiDataset(Dataset):
    '''
    adapted from: https://stackoverflow.com/questions/77528929/pytorch-imagefolder-vs-custom-dataset-from-single-folder
    '''
    def __init__(self, targ_dir, meta_file, transform=None):
        self.paths = list(Path(targ_dir).glob("*.png"))
        self.meta = pd.read_csv(meta_file)
        self.labels = list(map(self.get_label, self.paths))
        self.transform = transform
    
    def get_label(self, path):
        # make sure this function returns the label from the path
        match = re.search(r'images/(.+)', str(path)).group(1) # image ID
        label = self.meta.loc[self.meta['DDI_file'] == match]['malignant'].astype(bool) # malignant T/F label 
        label = int(label.iloc[0]) # pandas Series of len 1 to int
        return label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        class_idx = self.labels[index]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

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
    def __init__(self, latent_dim=20, num_classes=2, channels=3, width=299):
        super(ResNetSAE, self).__init__()
        self.latent_dim = latent_dim
        self.type = 'resnetsae'
        self.num_classes = num_classes
        
        # Load ResNet encoder
        resnet = models.resnet18(weights='IMAGENET1K_V1', progress=False)

        # Replace the fully connected layer
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_features, latent_dim),  # Map to latent_dim
        )
        # Encoder is ResNet
        self.encoder = resnet
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
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
            # Upsample: 256x256 -> 299x299
            nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        )

        # NN classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, num_classes)
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

def train_sae(model, train_data, val_data, epochs=50, lr=3e-4, recon_weight=0.5, task_weight=0.1):
    '''
    '''
    # init variables
    task_fn = nn.CrossEntropyLoss()
    decoder_fn = nn.MSELoss()
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    # begin training
    #print(f'Training {model.latent_dim}D model')
    #print('\tTraining Encoder, Decoder and Classifier')
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