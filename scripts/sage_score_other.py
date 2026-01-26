from prob_scoring import *
import argparse
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
import torchvision
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import sys
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        help="Type of encoder to use for SAGE model (one of ['ResNet', 'Inception', 'ViT'])."
    )
    parser.add_argument(
        "--dim",
        default=256,
        help="Dimensions of latent space (encoder output). Default is 32."
    )
    parser.add_argument(
        "--modeldir",
        help="Path to trained model directory."
    )
    parser.add_argument(
        "--datadir",
        help="Path to directory with subdirectories for image datasets."
    )
    parser.add_argument(
        "--compare",
        help="Name of imaging dataset to score against HAM10000."
    )
    parser.add_argument(
        "--outdir",
        help="Path to save embeddings and probability scores."
    )
    parser.add_argument(
        "--logdir",
        help="Path to folder for saving logs."
    )
    args = parser.parse_args()

    # set up output logging
    os.makedirs(args.logdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(f"{args.logdir}/sage_score_other.log"),
            logging.StreamHandler()  # still prints to console / docker logs
        ]
    )
    logger = logging.getLogger(__name__)

    assert args.encoder in ['ResNet', 'Inception', 'ViT']
    assert os.path.isdir(args.modeldir) # check model directory exists
    assert os.path.isdir(args.datadir) # check data directory exists
    assert os.path.isdir(os.path.join(args.datadir, args.compare)) # check subdir exists
    assert os.path.isdir(args.outdir) # check output directory exists

    dim = int(args.dim)
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
    
    # Split HAM10000 dataset into train (90%) and test (10%)
    ham_img_path = os.path.join(args.datadir, 'ham/images')
    ham_meta_path = os.path.join(args.datadir, 'ham/metadata.csv')
    ham_dataset = HamDataset(ham_img_path, ham_meta_path, transform=transform)
    
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
    # Further split into train and validation
    inner_splitter = StratifiedGroupKFold(
        n_splits=5, # 20% of 90% train split
        shuffle=True,
        random_state=33 # makes split reproducible
    )
    holder_labels = np.array(ham_dataset.labels)[holder_indices]
    holder_lesions = np.array(ham_dataset.lesions)[holder_indices]
    # gives first kfold split
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
    
    # Create subset for train
    train_dataset = Subset(ham_dataset, train_indices)

    batch_size = 256
    n_classes = 8

    # Create other dataset
    other_img_path = os.path.join(args.datadir, f'{args.compare}/images')
    other_meta_path = os.path.join(args.datadir, f'{args.compare}/metadata.csv')
    other_dataset = OtherDataset(other_img_path, other_meta_path, transform=transform)
    
    # Make data dict
    data_dict = dict()
    data_dict['train'] = train_dataset
    data_dict[f'{args.compare}'] = other_dataset
    
    ## Get SAGE output values for each of 5 models

    for i in range(5):
        suffix = f"{args.encoder}SAE_{dim}D_{i}.pth"
        model_path = os.path.join(args.modeldir, suffix)
        logger.info(f"Using model: {suffix}")
        # Init model
        if args.encoder == 'ResNet':
            model = ResNetSAE(latent_dim=dim, num_classes=8, channels=3)
        elif args.encoder == 'Inception':
            model = InceptionSAE(latent_dim=dim, num_classes=8, channels=3)
        elif args.encoder == 'ViT':
            model = VitSAE(latent_dim=dim, num_classes=8, channels=3)
        # Load model from save point
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(model_path, map_location=device)
        # Remove 'module.' if present from DataParallel wrapper during training
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(device) # run inference on single GPU
        
        # run SAGE
        latent_df = get_embedding(model, data_dict, dim, batch_size=batch_size, softmax=True)
        # get classifier confidence
        latent_df = get_max_conf(latent_df, 8)
        # get kNN distances to reference images
        latent_df = get_latent_distance(
            latent_df,
            data_dict.keys(),
            dim,
            k=20,
            reference='train',
            metric='manhattan'
        )
        # pickle latent df
        save_latent_path = os.path.join(args.outdir, f'{args.encoder}_{dim}D_{i}_latent_{args.compare}.pkl')
        with open(save_latent_path, 'wb') as file:
            pickle.dump(latent_df, file)
            
        # calculate probabilities and combined score
        probs_df = rank_measures_get_probs(
            latent_df,
            [dim],
            data_dict.keys(),
            reference='train'
        )
        # pickle probability df
        save_prob_path = os.path.join(args.outdir, f'{args.encoder}_{dim}D_{i}_probs_{args.compare}.pkl')
        with open(save_prob_path, 'wb') as file:
            pickle.dump(probs_df, file)

if __name__ == '__main__':
    main()