from model_setup import *
import re
import sys
import json
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--featureset",
    help="select GEXP model features to train SAGE: 'skgrid', 'aklimate', 'subscope' or 'all'"
)
args = parser.parse_args()

# only ['skgrid', 'aklimate', 'subscope'] models have GEXP features shared across TCGA, AURORA and METABRIC
with open('./tools/model_info.json', 'r') as fh:
    data = json.load(fh)
    if args.featureset == 'all':
        ft_list = []
        # add all eligible model features
        for mod in ['skgrid', 'aklimate', 'subscope']:
            ft_list.extend(data[mod]['BRCA']['GEXP']['fts'])
        ft_set = sorted(list(set(ft_list))) # removes duplicate features, sort ensures consistent order
    else:
        ft_set = data[args.featureset]['BRCA']['GEXP']['fts']

# read post-processed TCGA data
tcga_file = './data/TCGA_BRCA.tsv'
df = pd.read_csv(tcga_file, sep='\t', header=0)

# reassign labels
label_dict_tcga = {'BRCA_1':0, 'BRCA_2':1, 'BRCA_3':2, 'BRCA_4':3, 'BRCA_5':4, 'BRCA_6':5} # 4 and 5 for AURORA only
df['Labels'] = df['Labels'].replace(label_dict_tcga)

# split training data
X_train, X_val, y_train, y_val = train_test_split(
    df[ft_set],
    df['Labels'],
    test_size=0.15,
    stratify=df['Labels'],
    random_state=333
)
# init and fit simple scaler
scaler = MinMaxScaler()
scaler.fit(X_train)
# pickle scaler for subsequent use of trained SAGE model
with open(f'./models/scalers/{args.featureset}_scaler.pickle', 'wb') as file:
    pickle.dump(scaler, file)

# convert back to joint df for upscaling
train_df = pd.DataFrame(X_train, columns=ft_set)
train_df['Labels'] = y_train
# upsample minor classes to reduce imbalance
df_0 = train_df.loc[df['Labels'] == 0].copy() # 1x (from 454)
df_1 = pd.DataFrame(np.repeat(train_df.loc[train_df['Labels'] == 1], 2, axis=0), columns=train_df.columns) # 2x (to 348)
df_2 = pd.DataFrame(np.repeat(train_df.loc[train_df['Labels'] == 2], 2, axis=0), columns=train_df.columns) # 2x (to 298)
df_3 = pd.DataFrame(np.repeat(train_df.loc[train_df['Labels'] == 3], 5, axis=0), columns=train_df.columns) # 5x (to 340)
train_df = pd.concat([df_0, df_1, df_2, df_3])
X_train = train_df[ft_set]
y_train = train_df['Labels']

# Unpickle scaler
with open(f'./models/scalers/{args.featureset}_scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)
# scale features between 0 and 1
scaled_X_train = scaler.transform(X_train)
scaled_X_val = scaler.transform(X_val)

# train SAGE model
train_data = CustomTensorDataset(
    np.array(scaled_X_train).astype(np.float32),
    np.array(y_train).astype(np.int64)
)
model = FcSAE(in_features=len(ft_set))
trained_model, history = train_sae(model, train_data, epochs=50)
# save trained model
model_path = f'./models/{args.featureset}_sage'
torch.save(trained_model.state_dict(), model_path)

# calibrate model classifier
base_model = FcSAE(in_features=len(ft_set))
base_model.load_state_dict(torch.load(model_path)) # load model from save point
calib_model = ModelWithTemperature(base_model)
# tune temperature param using validation set
val_data = CustomTensorDataset(
    np.array(scaled_X_val).astype(np.float32),
    np.array(y_val).astype(np.int64)
)
val_dl = DataLoader(val_data, batch_size=64, shuffle=False)
set_temperature(calib_model, val_dl)
# save tempscaled model
wtemp_path = f'./models/{args.featureset}_sage_calib'
torch.save(calib_model.state_dict(), wtemp_path)

# check classification performance
dl = DataLoader(train_data, batch_size=64, shuffle=False)
preds = []
for data, labels in dl:
    _, _, logits = calib_model(data)
    preds.append(torch.argmax(logits, dim=1))
preds = [item for sublist in preds for item in sublist]
train_acc = accuracy_score(train_data.targets, preds)
train_acc = round(train_acc, 4)
print('\n')
print(f'Training accuracy: {train_acc}')

# check classification performance
dl = DataLoader(val_data, batch_size=64, shuffle=False)
preds = []
for data, labels in dl:
    _, _, logits = calib_model(data)
    preds.append(torch.argmax(logits, dim=1))
preds = [item for sublist in preds for item in sublist]
val_acc = accuracy_score(val_data.targets, preds)
val_acc = round(val_acc, 4)
print(f'Validation accuracy: {val_acc}')