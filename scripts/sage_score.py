from prob_scoring import *
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--featureset",
    help="select GEXP model features to train SAGE: 'skgrid', 'aklimate', 'subscope' or 'all'"
)
args = parser.parse_args()

## import TSV matrices
# read TCGA data
tcga_file = './data/TCGA_BRCA.tsv'
tcga = pd.read_csv(tcga_file, sep='\t', header=0)
# read METABRIC data
met_file = './data/METABRIC_BRCA.tsv'
metabric = pd.read_csv(met_file, sep='\t', header=0)
# read AURORA data
aur_file = './data/AURORA_BRCA.tsv'
aurora = pd.read_csv(aur_file, sep='\t', header=0)
# reassign labels
label_dict_tcga = {'BRCA_1':0, 'BRCA_2':1, 'BRCA_3':2, 'BRCA_4':3, 'BRCA_5':4, 'BRCA_6':5} # 4 and 5 for AURORA only
tcga['Labels'] = tcga['Labels'].replace(label_dict_tcga).copy()
metabric['Labels'] = metabric['Labels'].replace(label_dict_tcga).copy()
aurora['Labels'] = aurora['Labels'].replace(label_dict_tcga).copy()

## get feature names
# only ['skgrid', 'aklimate', 'subscope'] models have GEXP features shared across TCGA, AURORA and METABRIC
with open('./models/model_info.json', 'r') as fh:
    data = json.load(fh)
    if args.featureset == 'all':
        ft_list = []
        # add all eligible model features
        for mod in ['skgrid', 'aklimate', 'subscope']:
            ft_list.extend(data[mod]['BRCA']['GEXP']['fts'])
        ft_set = sorted(list(set(ft_list))) # removes duplicate features, sort ensures consistent order
    else:
        ft_set = data[args.featureset]['BRCA']['GEXP']['fts']

## Make datasets
# Unpickle scaler
with open(f'./models/scalers/{args.featureset}_scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)
# Scale features and make tensor datasets
scaled_tcga = scaler.transform(tcga[ft_set])
tcga_data = CustomTensorDataset(
    np.array(scaled_tcga).astype(np.float32),
    np.array(tcga['Labels']).astype(np.int64)
)
scaled_metabric = scaler.transform(metabric[ft_set])
metabric_data = CustomTensorDataset(
    np.array(scaled_metabric).astype(np.float32),
    np.array(metabric['Labels']).astype(np.int64)
)
scaled_aurora = scaler.transform(aurora[ft_set])
aurora_data = CustomTensorDataset(
    np.array(scaled_aurora).astype(np.float32),
    np.array(aurora['Labels']).astype(np.int64)
)
# make data dict
data_dict = dict()
data_dict['tcga'] = tcga_data
data_dict['metabric'] = metabric_data
data_dict['aurora'] = aurora_data

## Get SAGE output values
# load trained & calibrated model
model_path = f'./models/{args.featureset}_sage_calib'
# load best model parameters
#params_file = f'./models/{args.featureset}_train_history.pickle'
#with open(params_file, 'rb') as pf: 
    #model_params = pickle.load(pf)
#dims = model_params[min(list(model_params.keys()))]['latent_dim']
dims = 10
base_model = FcSAE(in_features=len(ft_set), latent_dim=dims)
calib_model = ModelWithTemperature(base_model)
calib_model.load_state_dict(torch.load(model_path)) # load model from save point
# run SAGE
latent_df = get_embedding(calib_model, data_dict, dims)
# get classifier confidence
latent_df = get_max_conf(latent_df, 4)
# get kNN distances to TCGA reference points
latent_df = get_latent_distance(
    latent_df,
    data_dict,
    dims,
    k=20,
    reference='tcga',
    metric='manhattan'
)
# calculate probabilities and combined score
probs_df = rank_measures_get_probs(
    latent_df,
    [dims],
    data_dict.keys(),
    reference='tcga'
)
# Add patient ID column and pickle & save outputs
id_df = pd.concat([tcga['BRCA'], metabric['BRCA'], aurora['BRCA']], axis=0).to_numpy()
latent_df['BRCA'] = id_df
probs_df['BRCA'] = id_df
# pickle latent df
with open(f'./data/sage_outputs/{args.featureset}_latent.pickle', 'wb') as file:
    pickle.dump(latent_df, file)
# pickle probability df
with open(f'./data/sage_outputs/{args.featureset}_probs.pickle', 'wb') as file:
    pickle.dump(probs_df, file)