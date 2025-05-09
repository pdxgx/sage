from prob_scoring import *
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        help="Type of encoder to use for SAGE model (one of ['ResNet', 'Inception', 'ViT'])."
    )
    parser.add_argument(
        "--dims",
        help="Number of encoder output dimensions for desired model."
    )
    parser.add_argument(
        "--modelpth",
        help="Path to trained HAM10000 SAGE model."
    )
    parser.add_argument(
        "--outdir",
        help="Path to save embeddings and probability scores."
    )
    args = parser.parse_args()

    assert args.encoder in ['ResNet', 'Inception', 'ViT']
    assert int(args.dims) in [2, 16, 32, 64]

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
    
    # Split dataset into train and test (90% train, 10% test)
    ham_img_path = '/home/groups/ThompsonLab/schreyer/sage/data/ham/images'
    ham_meta_path = '/home/groups/ThompsonLab/schreyer/sage/data/ham/metadata.csv'
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
    
    # Split dataset into train and test (90% train, 10% test)
    ddi_img_path = '/home/groups/ThompsonLab/schreyer/sage/data/ddi/images'
    ddi_meta_path = '/home/groups/ThompsonLab/schreyer/sage/data/ddi/metadata.csv'
    ddi_dataset = DdiDataset(ddi_img_path, ddi_meta_path, transform=transform)

    hiba_img_path = '/home/groups/ThompsonLab/schreyer/sage/data/hiba/images'
    hiba_meta_path = '/home/groups/ThompsonLab/schreyer/sage/data/hiba/metadata.csv'
    hiba_dataset = HibaDataset(hiba_img_path, hiba_meta_path, transform=transform)
    
    ufes_img_path = '/home/groups/ThompsonLab/schreyer/sage/data/ufes/images'
    ufes_meta_path = '/home/groups/ThompsonLab/schreyer/sage/data/ufes/metadata.csv'
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
    dims = int(args.dims)
    n_classes = 7

    # Init model
    if args.encoder == 'ResNet':
        model = ResNetSAE(latent_dim=dims, num_classes=n_classes, channels=3)
    elif args.encoder == 'Inception':
        model = InceptionSAE(latent_dim=dims, num_classes=n_classes, channels=3)
    elif args.encoder == 'ViT':
        model = VitSAE(latent_dim=dims, num_classes=n_classes, channels=3)
    # Load model from save point
    model.load_state_dict(torch.load(args.modelpth))
    
    # run SAGE
    latent_df = get_embedding(model, data_dict, dims, softmax=True)
    # get classifier confidence
    latent_df = get_max_conf(latent_df, n_classes)
    # get kNN distances to reference images
    latent_df = get_latent_distance(
        latent_df,
        data_dict.keys(),
        dims,
        k=25,
        reference='train',
        metric='manhattan'
    )
    # pickle latent df
    save_latent_path = os.path.join(args.outdir, f'all_{args.encoder}_{args.dims}D_latent.pkl')
    with open(save_latent_path, 'wb') as file:
        pickle.dump(latent_df, file)
        
    # calculate probabilities and combined score
    probs_df = rank_measures_get_probs(
        latent_df,
        [dims],
        data_dict.keys(),
        reference='train'
    )
    # pickle probability df
    save_prob_path = os.path.join(args.outdir, f'all_{args.encoder}_{args.dims}D_probs.pkl')
    with open(save_prob_path, 'wb') as file:
        pickle.dump(probs_df, file)

if __name__ == '__main__':
    main()