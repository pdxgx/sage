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
            logging.FileHandler(f"{args.logdir}/resnet_eval_other-{args.compare}.log"),
            logging.StreamHandler()  # still prints to console / docker logs
        ]
    )
    logger = logging.getLogger(__name__)

    assert os.path.isdir(args.modeldir) # check model directory exists
    assert os.path.isdir(args.datadir) # check data directory exists
    assert os.path.isdir(os.path.join(args.datadir, args.compare)) # check subdir exists
    assert os.path.isdir(args.outdir) # check output directory exists

    transform = transforms.Compose([
        transforms.Resize(299),               
        transforms.CenterCrop(299),            
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # use ImageNet values
    ])    

    batch_size = 256
    n_classes = 8

    # Create other dataset
    other_img_path = os.path.join(args.datadir, f'{args.compare}/images')
    other_meta_path = os.path.join(args.datadir, f'{args.compare}/metadata.csv')
    other_dataset = OtherDataset(other_img_path, other_meta_path, transform=transform)
    
    # Make data dict
    data_dict = dict()
    data_dict[f'{args.compare}'] = other_dataset
    
    # Deep ensemble results with metadata
    ens_df = eval_reset_ens(args.modeldir, data_dict, m=5, batch_size=batch_size) # uses all 5 trained ResNets
    # pickle and save ens df
    save_ens_path = os.path.join(args.outdir, f'resnet_ens_df_{args.compare}.pkl')
    with open(save_ens_path, 'wb') as file:
        pickle.dump(ens_df, file)

    # MC dropout with metadata
    mc_df = eval_resnet_mc(args.modeldir, data_dict, M=0, batch_size=batch_size) # uses first ResNet model by default
    # pickle and save mc df
    save_mc_path = os.path.join(args.outdir, f'resnet_mc_df_{args.compare}.pkl')
    with open(save_mc_path, 'wb') as file:
        pickle.dump(mc_df, file)

if __name__ == '__main__':
    main()