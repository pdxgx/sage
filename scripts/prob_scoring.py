from model_setup import *

def get_embedding(model, data_dict, dim, softmax=False):
    # init loss functions
    rloss_fxn = nn.MSELoss(reduction='none')
    sm = nn.Softmax(dim=1) # row-wise softmax
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # eval model for each dataset
    model = model.to(device).eval()
    # init defaultdict
    all_outputs = defaultdict(list)
    # loop datasets
    for name, data in data_dict.items():
        print(f'\tOutputting {name}')
        data_loader = DataLoader(data, batch_size=64, shuffle=False)
        # init per-batch holders
        batch_latents, batch_rloss, batch_logits, batch_labels, batch_ids = [], [], [], [], []
        with torch.no_grad():
            # get outputs
            for dat, lbls, ids in data_loader:
                dat, lbls, = dat.to(device), lbls.to(device) # push to device
                lbls = lbls.long() # ensures correct indexing
                encoded, decoded, logits = model(dat)
                
                # get per-sample reconstruction error
                rloss = rloss_fxn(decoded, dat).view(dat.size(0), -1).mean(dim=1)
                
                if softmax == True:
                    logits = sm(logits) # get softmax confidence scores when no calibration

                # Collect outputs
                batch_latents.append(encoded.cpu())
                batch_rloss.append(rloss.cpu())
                batch_logits.append(logits.cpu())
                batch_labels.append(lbls.cpu())
                batch_ids.extend(ids)
        
        # concat once per dataset
        all_outputs['latent'].append(torch.cat(batch_latents).numpy())
        all_outputs['rloss'].append(torch.cat(batch_rloss).numpy())
        all_outputs['logits'].append(torch.cat(batch_logits).numpy())
        all_outputs['labels'].append(torch.cat(batch_labels).numpy())
        all_outputs['ids'].extend(batch_ids)
        all_outputs['data'].extend([name] * len(data))
    
    # overall concat
    latent_array = np.concatenate(all_outputs['latent'])
    rloss_array = np.concatenate(all_outputs['rloss'])
    logits_array = np.concatenate(all_outputs['logits'])
    labels_array = np.concatenate(all_outputs['labels'])
    ids_array = np.array(all_outputs['ids'])
    data_array = np.array(all_outputs['data'])
    
    # make pandas df from SAGE measurements
    print('Making dataframe')
    latent_cols = ['latent'+str(i) for i in range(1, dim+1)] 
    conf_cols = ['conf'+str(i) for i in range(model.num_classes)]
    col_names = ['img_id', 'data', 'labels'] + latent_cols + ['rloss'] + conf_cols
    out_df = pd.DataFrame(columns=col_names)
    
    out_df['img_id'] = ids_array
    out_df['data'] = data_array
    out_df['labels'] = labels_array
    out_df['rloss'] = rloss_array
    
    # add confidence scores to df
    for i in range(model.num_classes):
        out_df[f'conf{i}'] = logits_array[:, i]
    # add latent embeddings to df
    for i in range(dim):
        out_df[f'latent{i+1}'] = latent_array[:, i]

    out_df['dim'] = dim
    
    return out_df

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

def compute_probabilities(points, reference):
    """
    Compute 1 - quantile (empirical CDF) of each point compared to sorted reference values.
    """
    reference = np.sort(reference)
    idx = np.searchsorted(reference, points)
    idx = np.clip(idx, 0, len(reference)) # limit indices between [0, len(ref)]
    return 1 - (idx / len(reference)) # one minus quantile probability
    
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
    
    results = defaultdict(list)

    for dim in latent_dims:
        dim_df = dataframe[dataframe.dim == dim]
        ref_df = dim_df[dim_df.data == reference].reset_index(drop=True)

        # get arrays for each SAE metric
        ref_metrics = {
            'dist': np.log(ref_df['dist'].to_numpy()),
            'rloss': ref_df['rloss'].to_numpy(),
            'task': ref_df['task'].to_numpy()
        }
        if task == 'classify':
            ref_metrics['task'] = -np.log(ref_metrics['task'])
        
        for name in datasets:
            name_df = dim_df[dim_df.data == name].reset_index(drop=True)

            if local:
                if indices is None or not any(col.startswith('result') for col in indices.columns):
                        raise NameError(
                            'Missing kNN index columns. Pass output of "fit_tree_get_measures" with "return_indices=True".'
                        )
                index_df = indices[indices.data == name]
                knn_indices = index_df[[col for col in index_df.columns if col.startswith('result')]].to_numpy()
    
                for metric in ['dist', 'rloss', 'task']:
                    ref_values = ref_metrics[metric][knn_indices]
                    if metric == 'dist':
                        values = np.log(name_df[metric].to_numpy())
                    elif metric == 'task' and task == 'classify':
                        values = -np.log(name_df[metric].to_numpy())
                    else:
                        values = name_df[metric].to_numpy()
    
                    probs = np.array([
                        compute_probabilities(np.array([v]), ref_row)[0]
                        for v, ref_row in zip(values, ref_values)
                    ])
                    results[f'{metric}_prob'].append(probs)
            
            else:
                for metric in ['dist', 'rloss', 'task']:
                    if metric == 'dist':
                        values = np.log(name_df[metric].to_numpy())
                    elif metric == 'task' and task == 'classify':
                        values = -np.log(name_df[metric].to_numpy())
                    else:
                        values = name_df[metric].to_numpy()

                    probs = compute_probabilities(values, ref_metrics[metric])
                    results[f'{metric}_prob'].append(probs)

            # append metadata
            results['img_id'].append(name_df['img_id'].to_numpy())
            results['data'].append([name] * len(name_df))
            results['labels'].append(name_df['labels'].to_numpy())
            results['dim'].append([dim] * len(name_df))
        
    # make output df
    probs_df = pd.DataFrame({
        'img_id': np.concatenate(results['img_id']),
        'data': np.concatenate(results['data']),
        'labels': np.concatenate(results['labels']),
        'dim': np.concatenate(results['dim']),
        'dist_prob': np.concatenate(results['dist_prob']),
        'rloss_prob': np.concatenate(results['rloss_prob']),
        'task_prob': np.concatenate(results['task_prob']),
    })
    
    # geometric mean of the output probabilities is the combined score
    probs_df['gmean_metric'] = (probs_df['dist_prob'] * probs_df['rloss_prob'] * probs_df['task_prob']) ** (1/3)
    
    return probs_df