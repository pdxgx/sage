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
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

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
        "--logdir",
        help="Path to folder for saving stdout logs."
    )
    parser.add_argument(
        "--balanced",
        action='store_true',
        help="Uses balanced batch sampling during model training."
    )
    args = parser.parse_args()
    
    # set up output logging
    os.makedirs(args.logdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(f"{args.logdir}/train_ham_resnets.log"),
            logging.StreamHandler()  # still prints to console / docker logs
        ]
    )
    logger = logging.getLogger(__name__)

    # check args
    assert os.path.isdir(args.imagedir) # check image directory exists
    assert os.path.isfile(args.metafile) # check metadata file exists
    assert os.path.isdir(args.savedir) # check save directory exists
    assert os.path.isdir(args.logdir) # check log directory exists
    assert args.balanced in [True, False]
    
    # Transform used to train DeepDerm model (based on Inception V3) in DDI paper, missing cutout of upright rectangle
    paper_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(0, 359)),    
        transforms.RandomVerticalFlip(p=0.5),       
        transforms.Resize(299),                 
        transforms.CenterCrop(299),            
        transforms.ToTensor(),                  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform = transforms.Compose([
        transforms.Resize(299),               
        transforms.CenterCrop(299),            
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # use ImageNet values
    ])
    
    # Split dataset into train (90%) and test (10%)
    ham_dataset = HamDataset(args.imagedir, args.metafile, transform=None)
    splitter = StratifiedGroupKFold(
        n_splits=10, # 10% test split
        shuffle=True,
        random_state=33 # makes split reproducible
    )
    # gives first kfold split
    holder_indices, test_indices = next(
        splitter.split(
            np.zeros(len(ham_dataset)), # stand-in for X
            ham_dataset.labels, # holds class ratios constant
            ham_dataset.lesions # separates by lesion ID
        )
    )
    # Further split
    inner_splitter = StratifiedGroupKFold(
        n_splits=5, # 20% of 90% validation split
        shuffle=True,
        random_state=33 # makes split reproducible
    )
    # placeholder arrays for second split
    holder_labels = np.array(ham_dataset.labels)[holder_indices]
    holder_lesions = np.array(ham_dataset.lesions)[holder_indices]
    # splits train into train/valid
    train_indices, val_indices = next(
        inner_splitter.split(
            np.zeros(len(holder_indices)), # stand-in for X
            holder_labels, # holds class ratios constant
            holder_lesions # separates by lesion ID
        )
    )
    # map holder indices back to original dataset
    train_indices = holder_indices[train_indices]
    val_indices = holder_indices[val_indices]
    
    # ensure no patient leakage
    assert len(set(train_indices).intersection(set(test_indices))) == 0 # check no overlap between train/test
    assert len(set(train_indices).intersection(set(val_indices))) == 0 # check no overlap between train/val
    assert len(set(val_indices).intersection(set(test_indices))) == 0 # check no overlap between val/test
    
    # Init individual datasets
    ham_train = HamDataset(args.imagedir, args.metafile, transform=transform)
    ham_val = HamDataset(args.imagedir, args.metafile, transform=transform)
    ham_test = HamDataset(args.imagedir, args.metafile, transform=transform)
    
    # Create subsets for train, test and valid
    train_dataset = Subset(ham_train, train_indices)
    test_dataset = Subset(ham_test, test_indices)
    val_dataset = Subset(ham_val, val_indices)
    
    # Init training params
    batch_size = 64 # must be divisible by n classes for balanced sampler
    train_labels = [ham_dataset.labels[i] for i in train_indices] # returns labels for train images
    
    if args.balanced == True:
        sampler = BalancedBatchSampler(train_labels, batch_size=batch_size)
        weights = None
        path_suf = '_balanced'
    else:
        sampler = None
        weights = get_train_weights(train_labels)
        path_suf = ''

    # train 5 ResNets
    for i in range(5):
        model = ResNet(num_classes=8)
        
        # Run training loop
        model, history = train_resnet(
            model, 
            train_dataset, 
            val_dataset, 
            epochs=50,
            batch_size=batch_size,
            sampler=sampler,
            weights=weights
        )
        
        # Save trained model as state dict
        model_path = os.path.join(args.savedir, f"{get_model_component(model, 'type')}_{i}{path_suf}.pth")
        torch.save(get_state_dict(model), model_path)
        
        # Save history as pickled object
        hist_path = os.path.join(args.savedir, f"{get_model_component(model, 'type')}_{i}{path_suf}_history.pkl")
        with open(hist_path, 'wb') as file:
            pickle.dump(history, file)

if __name__ == '__main__':
    main()
