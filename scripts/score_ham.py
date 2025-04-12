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
    parser.add_argument(
        "--outdir",
        help="Path to save HAM10000 train and test embeddings and probability scores."
    )
    args = parser.parse_args()

    # Load train and test data
    transform = transforms.Compose([
        transforms.Resize(320),               
        transforms.CenterCrop(299),            
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Split dataset into train and test (80% train, 20% test)
    all_dataset = DermoDataset(args.imagedir, args.metafile, transform=transform)
    train_indices, test_indices = train_test_split(
        np.arange(len(all_dataset)),
        test_size=0.2,
        stratify=all_dataset.labels,
        random_state=33
    )
    # Create subsets for train and test
    train_dataset = Subset(all_dataset, train_indices)
    test_dataset = Subset(all_dataset, test_indices)
    # Make data dict
    data_dict = dict()
    data_dict['train'] = train_dataset
    data_dict['test'] = test_dataset
    
    ## Get SAGE output values
    # load trained model
    dims = 20
    model = ResNetSAE(latent_dim=dims, num_classes=2, channels=3, width=299)#.cuda()
    model.load_state_dict(torch.load(args.modelpth)) # load model from save point
    # run SAGE
    latent_df = get_embedding(model, data_dict, dims, softmax=True)
    # get classifier confidence
    latent_df = get_max_conf(latent_df, 2)
    # get kNN distances to TCGA reference points
    latent_df = get_latent_distance(
        latent_df,
        data_dict.keys(),
        dims,
        k=20,
        reference='train',
        metric='manhattan'
    )
    # pickle latent df
    save_latent_path = os.path.join(args.outdir, 'ham_resnetsae_latent.pickle')
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
    save_prob_path = os.path.join(args.outdir, 'ham_resnetsae_probs.pickle')
    with open(save_prob_path, 'wb') as file:
        pickle.dump(probs_df, file)

if __name__ == '__main__':
    main()