from model_setup import *
import torch
from torchvision import datasets
from torchvision.transforms import v2
import numpy as np
import os
import gzip
import pickle
import subprocess
import glob
import random

def get_embedding(model, data_dict, dim, softmax=False):
    # init holder lists
    latent_list, rloss_list, task_list, name_list, label_list = [], [], [], [], []
    # init loss functions
    rloss_fxn = nn.MSELoss()
    sm = nn.Softmax(dim=1) # row-wise softmax
    # eval model for each dataset
    model.eval()
    for name, data in data_dict.items():
        print(f'\tOutputting {name}')
        # get output
        data_loader = DataLoader(data, batch_size=64, shuffle=False)
        latent_space, rloss, task, labels  = [], [], [], []
        with torch.no_grad():
            for dat, lbls in data_loader:
                encoded, decoded, logits = model(dat)
                latent_space.append(encoded)
                # get reconstruction error
                for dec, d in zip(decoded, dat):
                    r_loss = rloss_fxn(dec, d)
                    rloss.append(r_loss)
                if softmax == True:
                    logits = sm(logits) # get softmax confidence scores when no calibration
                task.append(logits)
                labels.append(lbls)
        latent_space = torch.cat(latent_space).numpy()
        rloss = np.array(rloss)
        task = torch.cat(task).numpy()
        labels = torch.cat(labels).numpy()
        # append to lists
        latent_list.append(latent_space)
        rloss_list.append(rloss)
        task_list.append(task)
        label_list.append(labels)
        name_list.append([name]*len(data_dict[name]))
        #try:
            #targets = data.targets
        #except AttributeError:
            #targets = data.labels
        #label_list.append(np.array(targets))
    
    # make pandas df from SAE outputs
    print('Making dataframe')
    latent_cols = ['latent'+str(i) for i in range(1, dim+1)] 
    conf_cols = ['conf'+str(i) for i in range(model.num_classes)]
    col_names = ['data', 'labels'] + latent_cols + ['rloss'] + conf_cols
    latent_df = pd.DataFrame(columns=col_names)
    # add confidence scores to df
    for i in range(model.num_classes):
        conf = np.concatenate([j[:,i] for j in task_list], axis=None)
        latent_df[f'conf{i}'] = conf
    # add latent embeddings to df
    for i in range(dim):
        # get latent embedding
        embed = np.concatenate([j[:,i] for j in latent_list], axis=None)
        # add to df
        latent_df[f'latent{i+1}'] = embed
    
    # unroll name and rloss lists and add to df
    latent_df['data'] = np.concatenate(name_list, axis=None)
    latent_df['labels'] = np.concatenate(label_list, axis=None)
    latent_df['rloss'] = np.concatenate(rloss_list, axis=None)
    latent_df['dim'] = [dim]*len(latent_df)
    
    return latent_df

def get_regression_embedding(model, data_dict, dim):
    # init holder lists
    latent_list, rloss_list, task_list, name_list, label_list = [], [], [], [], []
    # init loss functions
    rloss_fxn = nn.MSELoss()
    tloss_fxn = nn.MSELoss()
    # eval model for each dataset
    model.eval()
    for name, data in data_dict.items():
        print(f'\tOutputting {name}')
        # get output
        data_loader = DataLoader(data, batch_size=64, shuffle=False)
        latent_space, rloss, task  = [], [], []
        with torch.no_grad():
            for dat, lbls in data_loader:
                encoded, decoded, logits = model(dat)
                latent_space.append(encoded)
                # get reconstruction error
                for dec, d in zip(decoded, dat):
                    r_loss = rloss_fxn(dec, d)
                    rloss.append(r_loss)
                # get regression error
                logits = logits.view(lbls.size(0))
                for log, lbl in zip(logits, lbls):
                    t_loss = tloss_fxn(log, lbl)
                    task.append(t_loss)
        latent_space = torch.cat(latent_space).numpy()
        rloss = np.array(rloss)
        task = np.array(task)
        # append to lists
        latent_list.append(latent_space)
        rloss_list.append(rloss)
        task_list.append(task)
        name_list.append([name]*len(data_dict[name]))
        try:
            targets = data.targets
        except AttributeError:
            targets = data.labels
        label_list.append(np.array(targets))
    
    # make pandas df from SAE outputs
    print('Making dataframe')
    latent_cols = ['latent'+str(i) for i in range(1, dim+1)] 
    col_names = ['data', 'labels'] + latent_cols + ['rloss', 'task']
    latent_df = pd.DataFrame(columns=col_names)
    latent_df['task'] = np.concatenate(task_list, axis=None)
    # add latent embeddings to df
    for i in range(dim):
        # get latent embedding
        embed = np.concatenate([j[:,i] for j in latent_list], axis=None)
        # add to df
        latent_df[f'latent{i+1}'] = embed
    
    # unroll name and rloss lists and add to df
    latent_df['data'] = np.concatenate(name_list, axis=None)
    latent_df['labels'] = np.concatenate(label_list, axis=None)
    latent_df['rloss'] = np.concatenate(rloss_list, axis=None)
    latent_df['dim'] = [dim]*len(latent_df)
    
    return latent_df

def get_max_conf(latent_df, n_classes):
    conf_cols = ['conf'+str(i) for i in range(n_classes)]
    task_data = latent_df[conf_cols]
    latent_df['task'] = task_data.max(axis=1) # maximum confidence value for rows
    return latent_df

def get_latent_distance(latent_df, datasets, dim, k=100, reference='train', metric='manhattan'):
    dist_arr = []
    latent_cols = ['latent'+str(i) for i in range(1, dim+1)]
    # get training latent space
    train_latent = latent_df.loc[latent_df['data'] == reference][latent_cols].to_numpy()
    # fit BallTree to train latent space, use L1 distance metric as default
    tree = BallTree(train_latent, metric=metric)
    for name in datasets:
        name_latent = latent_df.loc[latent_df['data'] == name][latent_cols].to_numpy()
        # query tree
        if name == reference:
            distances, indices = tree.query(name_latent, k=k+1)
            # removes columns where train points return themselves (used to fit tree)
            distances = np.delete(distances, 0, 1)
            indices = np.delete(indices, 0, 1)
        else:
            distances, indices = tree.query(name_latent, k=k)
        # average kNN distances
        distances = np.mean(distances, axis=1) # correct axis?
        dist_arr.append(distances)
    latent_df['dist'] = np.concatenate(dist_arr, axis=None)

    return latent_df

def rank_measures_get_probs(dataframe, latent_dims, datasets, reference='train', task='classify', local=False, indices=None):
    '''
    input:
        (pd.DataFrame) output of 'get_latent_distance',
        (list) dimensions of encoder output,
        (list) names of datasets used to query kNN tree,
        (str) name of the dataset to use for reference SAE measure arrays,
        (str) "classify" or "regress",
        (bool) whether to use kNN measures of distance, reconstruction error and classifier confidence to calculate probability
        (pd.DataFrame) output of 'fit_tree_get_measures', indices for reference points returned from kNN query
    '''
    warnings.filterwarnings("error", category=RuntimeWarning) # catch as errors for try, except clause
    assert task in ["classify", "regress"]
    assert local in [True, False]
    if local == True:
        if len([col for col in indices.columns if col.startswith('result')]) == 0:
            raise NameError(
                'Attempting to use local results for scoring with no kNN index columns.'
                'Pass output of "fit_tree_get_measures" with "return_indices = True" to "indices" argument.'
            )
    dim_arr = []
    for dim in latent_dims:
        dim_df = dataframe.loc[dataframe.dim == dim]
        dim_arr.append([dim] * len(dim_df))
        # using 'train' dataset as reference unless specified
        ref_df = dim_df.loc[dim_df.data == reference].reset_index(drop=True)
        # get arrays for each SAE metric
        ref_dist = np.log(ref_df['dist'].to_numpy())
        ref_rloss = ref_df['rloss'].to_numpy()
        ref_task = ref_df['task'].to_numpy()
        if task == 'classify':
            ref_task = -np.log(ref_task)
        
        if local == True:
            name_arr, label_arr, dist_prob, rloss_prob, task_prob = [], [], [], [], [] # placeholder arrays
            result_cols = [col for col in indices.columns if col.startswith('result')] # how we identify kNN query result columns
            for name in datasets: # find reference probability for each binned SAE metric
                # get name copies of dfs
                index_df = indices.loc[indices.data == name]
                name_df = dim_df.loc[dim_df.data == name]
                # replace returned kNN indices with ref measures
                returned_dist = ref_dist[index_df[result_cols].to_numpy()]
                returned_rloss = ref_rloss[index_df[result_cols].to_numpy()]
                returned_task = ref_task[index_df[result_cols].to_numpy()]
                # find probability of dataset measures using local kNN bins
                for meas in ['dist', 'rloss', 'task']:
                    ref_array = eval('returned_' + meas)
                    ref_array = np.sort(ref_array, axis=1) # sort returned values in each row
                    name_measure = name_df[meas].to_numpy()
                    if meas == 'dist':
                        name_measure = np.log(name_measure)
                    elif meas == 'task' and task == 'classify':
                        name_measure = -np.log(name_measure)
                    for row, point in enumerate(name_measure): # loop through each measure
                        ref = ref_array[row] # get returned index values for row
                        index = np.searchsorted(ref, point)
                        index = np.clip(index, 0, len(ref)) # limit indices between [0, len(ref)]
                        prob = 1 - (index / len(ref)) # one minus quantile probability
                        eval(meas + '_prob').append(prob)
                name_arr.append([name]*len(name_df))
                label_arr.append(name_df['labels'].to_numpy())
                    
        else:
            name_arr, label_arr, dist_prob, rloss_prob, task_prob = [], [], [], [], [] # placeholder arrays
            for name in datasets: 
                name_df = dim_df.loc[dim_df.data == name]
                for meas in ['dist', 'rloss', 'task']: # find probability using each reference SAE metric
                    meas_array = name_df[meas].to_numpy()
                    ref_array = eval('ref_' + meas)
                    if meas == 'task' and task == 'classify': # neg log transform of confidence values reorients array where lower is better
                        meas_array = -np.log(meas_array)
                    elif meas == 'dist':
                        meas_array = np.log(meas_array)
                    ref_array = np.sort(ref_array) # sort reference measures low -> high
                    for point in meas_array:
                        index = np.searchsorted(ref_array, point)
                        index = np.clip(index, 0, len(ref_array)) # limit indices between [0, len(ref_array)]
                        # one minus quantile position
                        prob = 1 - (index / len(ref_array))
                        if name == 'train' and prob == 0:
                            print(f'train {meas} probability is 0.0 for {point}')
                        eval(meas + '_prob').append(prob)
                name_arr.append([name]*len(name_df))
                label_arr.append(name_df['labels'].to_numpy())
    
    # make output df
    probs_df = pd.DataFrame(
        {
            'data':np.concatenate(name_arr, axis=None),
            'dim':np.concatenate(dim_arr, axis=None),
            'labels':np.concatenate(label_arr, axis=None),
            'dist_prob':np.concatenate(dist_prob, axis=None),
            'rloss_prob':np.concatenate(rloss_prob, axis=None),
            'task_prob':np.concatenate(task_prob, axis=None)
        }
    )
    # geometric mean of the output probabilities is the combined score
    probs_df['gmean_metric'] = (probs_df['dist_prob'] * probs_df['rloss_prob'] * probs_df['task_prob']) ** (1/3)
    
    return probs_df

def make_abalone_preds_df(probs_df, data_dict, original_columns):
    name_arr = []
    data_arr, target_arr = [], []
    for name in probs_df['data'].unique():
        data_arr.append(data_dict[name].data.numpy())
        target_arr.append(data_dict[name].targets.numpy())
        name_arr.append([name]*len(data_dict[name].data))
    preds_df = pd.DataFrame(np.vstack(data_arr), columns=original_columns)
    preds_df['Rings'] = np.concatenate(target_arr, axis=None)
    preds_df['data'] = np.concatenate(name_arr, axis=None)
    preds_df['gmean_metric'] = probs_df['gmean_metric']
    
    return preds_df