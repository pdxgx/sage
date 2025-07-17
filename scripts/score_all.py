from prob_scoring import *
import argparse
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
import torchvision
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        help="Type of encoder to use for SAGE model (one of ['ResNet', 'Inception', 'ViT'])."
    )
    parser.add_argument(
        "--dim",
        default=32,
        help="Dimensions of latent space (encoder output). Default is 32."
    )
    parser.add_argument(
        "--modelpth",
        help="Path to trained HAM10000 SAGE model."
    )
    parser.add_argument(
        "--datadir",
        help="Path to dataset directory with subdirectories for HAM10000, HIBA, UFES and DDI images."
    )
    parser.add_argument(
        "--outdir",
        help="Path to save embeddings and probability scores."
    )
    args = parser.parse_args()

    assert args.encoder in ['ResNet', 'Inception', 'ViT']
    assert os.path.isfile(args.modelpath) # check model exists
    assert os.path.isdir(args.datadir) # check data directory exists
    assert os.path.isdir(args.outdir) # check outdir exists

    if type(args.dim) != int:
        dim == int(args.dim)
    else:
        dim = args.dim

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
    
    # Split dataset into train (90%) and test
    ham_img_path = os.path.join(args.datadir, '/ham/images')
    ham_meta_path = os.path.join(args.datadir, '/ham/metadata.csv')
    ham_dataset = HamDataset(ham_img_path, ham_meta_path, transform=transform)
    train_indices, test_indices = train_test_split(
        np.arange(len(ham_dataset)),
        test_size=0.1,
        stratify=ham_dataset.labels,
        random_state=33 # makes split reproducible
    )
    # Create subsets for train and test
    train_dataset = Subset(ham_dataset, train_indices)
    test_dataset = Subset(ham_dataset, test_indices)
    
    # Create other datasets
    ddi_img_path = os.path.join(args.datadir, '/ddi/images')
    ddi_meta_path = os.path.join(args.datadir, '/ddi/metadata.csv')
    ddi_dataset = DdiDataset(ddi_img_path, ddi_meta_path, transform=transform)

    hiba_img_path = os.path.join(args.datadir, '/hiba/images')
    hiba_meta_path = os.path.join(args.datadir, '/hiba/metadata.csv')
    hiba_dataset = HibaDataset(hiba_img_path, hiba_meta_path, transform=transform)
    
    ufes_img_path = os.path.join(args.datadir, '/ufes/images')
    ufes_meta_path = os.path.join(args.datadir, '/ufes/metadata.csv')
    ufes_dataset = UfesDataset(ufes_img_path, ufes_meta_path, transform=transform)
    
    # Make data dict
    data_dict = dict()
    data_dict['train'] = train_dataset
    data_dict['test'] = test_dataset
    data_dict['ddi'] = ddi_dataset
    data_dict['hiba'] = hiba_dataset
    data_dict['ufes'] = ufes_dataset
    
    ## Get SAGE output values
    
    # load trained model
    n_classes = 7 # uses HAM10000 classes

    # Init model
    if args.encoder == 'ResNet':
        model = ResNetSAE(latent_dim=dim, num_classes=n_classes, channels=3)
    elif args.encoder == 'Inception':
        model = InceptionSAE(latent_dim=dim, num_classes=n_classes, channels=3)
    elif args.encoder == 'ViT':
        model = VitSAE(latent_dim=dim, num_classes=n_classes, channels=3)
    # Load model from save point
    model.load_state_dict(torch.load(args.modelpth))
    
    # run SAGE
    latent_df = get_embedding(model, data_dict, dim, softmax=True)
    # get classifier confidence
    latent_df = get_max_conf(latent_df, n_classes)
    # get kNN distances to reference images
    latent_df = get_latent_distance(
        latent_df,
        data_dict.keys(),
        dim,
        k=25,
        reference='train',
        metric='manhattan'
    )
    # pickle latent df
    save_latent_path = os.path.join(args.outdir, f'all_{args.encoder}_{dim}D_latent.pkl')
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
    save_prob_path = os.path.join(args.outdir, f'all_{args.encoder}_{dim}D_probs.pkl')
    with open(save_prob_path, 'wb') as file:
        pickle.dump(probs_df, file)

if __name__ == '__main__':
    main()