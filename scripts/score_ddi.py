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
        help="Directory with DDI dataset images."
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
        help="Path to save DDI embeddings and probability scores."
    )
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize(320),               
        transforms.CenterCrop(299),            
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load DDI data as dataset
    ddi_dataset = DdiDataset(args.imagedir, args.metafile, transform=transform)
    # Make data dict
    data_dict = dict()
    data_dict['ddi'] = ddi_dataset
    
    ## Get SAGE output values
    # load trained model
    dims = 20
    model = ResNetSAE(latent_dim=dims, num_classes=2, channels=3, width=299)#.cuda()
    model.load_state_dict(torch.load(args.modelpth)) # load model from save point
    # run SAGE
    ddi_latent = get_embedding(model, data_dict, dims, softmax=True) # no softmax applied in forward method
    # get classifier confidence
    ddi_latent = get_max_conf(ddi_latent, 2)
    
    # load HAM10000 data
    ham_latent_file = '/Users/schreyer/scratch/sage/data/sage_outputs/ham_resnetsae_latent.pickle'
    with open(ham_latent_file, 'rb') as hl:
        ham_latent = pickle.load(hl)

    ddi_ham_latent = pd.concat([ham_latent, ddi_latent])
    dataset_names = ['train', 'test', 'ddi']
    # get kNN distances to TCGA reference points
    ddi_ham_latent = get_latent_distance(
        ddi_ham_latent,
        dataset_names,
        dims,
        k=20,
        reference='train',
        metric='manhattan'
    )
    # pickle latent df
    save_latent_path = os.path.join(args.outdir, 'ddi_resnetsae_latent.pickle')
    with open(save_latent_path, 'wb') as file:
        pickle.dump(ddi_ham_latent, file)

    # calculate probabilities and combined score
    probs_df = rank_measures_get_probs(
        ddi_ham_latent,
        [dims],
        dataset_names,
        reference='train'
    )
    # pickle probability df
    save_prob_path = os.path.join(args.outdir, 'ddi_resnetsae_probs.pickle')
    with open(save_prob_path, 'wb') as file:
        pickle.dump(probs_df, file)

if __name__ == '__main__':
    main()