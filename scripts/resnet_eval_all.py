from prob_scoring import *
import argparse
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
import torchvision
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modeldir",
        help="Path to trained model directory."
    )
    parser.add_argument(
        "--datadir",
        help="Path to directory with subdirectories for image datasets."
    )
    parser.add_argument(
        "--outdir",
        help="Path to save embeddings and probability scores."
    )
    parser.add_argument(
        "--logdir",
        help="Path to folder for saving logs."
    )
    parser.add_argument(
        "--balanced",
        action='store_true',
        help="Load model with '_balanced' suffix in pathname."
    )
    args = parser.parse_args()

    # set up output logging
    os.makedirs(args.logdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(f"{args.logdir}/resnet_eval_all.log"),
            logging.StreamHandler()  # still prints to console / docker logs
        ]
    )
    logger = logging.getLogger(__name__)

    assert os.path.isdir(args.modeldir) # check model directory exists
    assert os.path.isdir(args.datadir) # check data directory exists
    assert os.path.isdir(args.outdir) # check output directory exists

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
    
    # Create subsets for train, test and valid
    train_dataset = Subset(ham_dataset, train_indices)
    test_dataset = Subset(ham_dataset, test_indices)

    batch_size = 256
    n_classes = 8
    
    # Create other datasets
    ddi_img_path = os.path.join(args.datadir, 'ddi/images')
    ddi_meta_path = os.path.join(args.datadir, 'ddi/metadata.csv')
    ddi_dataset = DdiDataset(ddi_img_path, ddi_meta_path, transform=transform)

    hiba_img_path = os.path.join(args.datadir, 'hiba/images')
    hiba_meta_path = os.path.join(args.datadir, 'hiba/metadata.csv')
    hiba_dataset = IsicDataset(hiba_img_path, hiba_meta_path, transform=transform)
    
    ufes_img_path = os.path.join(args.datadir, 'ufes/images')
    ufes_meta_path = os.path.join(args.datadir, 'ufes/metadata.csv')
    ufes_dataset = UfesDataset(ufes_img_path, ufes_meta_path, transform=transform)

    milk_img_path = os.path.join(args.datadir, 'milk10k/images')
    milk_meta_path = os.path.join(args.datadir, 'milk10k/metadata.csv')
    milk_dataset = OtherDataset(milk_img_path, milk_meta_path, transform=transform)

    caltech_img_path = download_caltech101(args.datadir)
    caltech_dataset = Caltech101Dataset(root=caltech_img_path, transform=transform)
    
    # Make data dict
    data_dict = dict()
    data_dict['train'] = train_dataset
    data_dict['test'] = test_dataset
    data_dict['ddi'] = ddi_dataset
    data_dict['hiba'] = hiba_dataset
    data_dict['ufes'] = ufes_dataset
    data_dict['milk10k'] = milk_dataset
    data_dict['cal101'] = caltech_dataset

    if args.balanced:
        balanced = True
        path_suf = '_balanced'
    else:
        balanced = False
        path_suf = ''
    
    # Deep ensemble results with metadata
    ens_df = eval_reset_ens(args.modeldir, data_dict, m=5, batch_size=batch_size, balanced=balanced) # uses all 5 trained ResNets
    # pickle and save ens df
    save_ens_path = os.path.join(args.outdir, f'resnet_ens{path_suf}_df.pkl')
    with open(save_ens_path, 'wb') as file:
        pickle.dump(ens_df, file)

    # MC dropout with metadata
    mc_df = eval_resnet_mc(args.modeldir, data_dict, M=0, batch_size=batch_size, balanced=balanced) # uses first ResNet model by default
    # pickle and save mc df
    save_mc_path = os.path.join(args.outdir, f'resnet_mc{path_suf}_df.pkl')
    with open(save_mc_path, 'wb') as file:
        pickle.dump(mc_df, file)

if __name__ == '__main__':
    main()