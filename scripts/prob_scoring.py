from model_setup import *

# init logging
logger = logging.getLogger(__name__)

def eval_reset_ens(modeldir, data_dict, m=5, batch_size=128, balanced=False):
    """
    Evaluates m trained ResNet models on pytorch image datasets and returns predictions.
    
    Args:
        modeldir: Directory holding trained models
        data_dict: Dictionary holding pytorch datasets
        m: (int) Number of trained models to use
    
    Returns:
        out_df: pandas df with row for each image and the following columns:
            img_id, labels, dataset, probs (list of m lists), mean prediction, std of predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probs_all = []
    if balanced == True:
        path_suf = '_balanced'
    else:
        path_suf = ''
    for j in range(m):
        model = ResNet().to(device) # runs on single GPU
        # Get model weights
        resnet_path = os.path.join(modeldir, f"ResNet_{j}{path_suf}.pth")
        logger.info(f"Using model: ResNet_{j}{path_suf}.pth | {j+1}/{m}")
        state = torch.load(resnet_path, map_location=device)
        # Remove 'module.' if present from DataParallel wrapper during training
        state = {k.replace("module.", ""): v for k, v in state.items()}
        # Load from save point
        model.load_state_dict(state)
        
        # init probs holder
        model_probs = []
        # init metadata holders once
        if j == 0:
            labels_all = []
            img_ids_all = []
            datasets_all = []
        
        model.eval()
        with torch.inference_mode():
            for name, data in data_dict.items():
                logger.info(f'\tOutputting {name}')
                data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
                # init per-batch holders
                batch_probs, batch_labels, batch_ids = [], [], []
                # get outputs
                for dat, lbs, ids in data_loader:
                    dat = dat.to(device, non_blocking=True) # push to device
                    logits = model(dat)
                    probs = F.softmax(logits, dim=-1)
                    model_probs.append(probs.cpu())
                    
                    # get metadata once
                    if j == 0:
                        labels_all.append(lbs.numpy())
                        img_ids_all.extend(ids)
                        datasets_all.extend([name] * len(ids))
                    
            # concat probabilies once per model
            model_probs = torch.cat(model_probs)
            probs_all.append(model_probs)
            
            # collect metadata once  
            if j == 0:
                labels_all = np.concatenate(labels_all)
                img_ids_all = np.array(img_ids_all)
                datasets_all = np.array(datasets_all)
        
    # overall concat
    probs_all = torch.stack(probs_all).cpu().numpy() # shape: (m, N, n_classes)
    probs_per_image = np.moveaxis(probs_all, 0, 1) # refactors shape so rows are convunique to image: (N, m, n_classes)

    out_df = pd.DataFrame({
        "img_id": img_ids_all,
        "labels": labels_all,
        "data": datasets_all,
        "probs": list(probs_per_image), # shape: (m, n_classes)
    })
    out_df["ens_mean"] = out_df["probs"].apply(lambda x: x.mean(axis=0))
    out_df["ens_std"] = out_df["probs"].apply(lambda x: x.std(axis=0))
    
    return out_df

def eval_resnet_mc(modeldir, data_dict, M=0, batch_size=128, balanced=False):
    """
    Evaluates trained ResNet models on pytorch image datasets with MC dropout forward pass and returns predictions.
    
    Args:
        modeldir: Directory holding trained models
        data_dict: Dictionary holding pytorch datasets
        M (int): ID of trained model to use
    
    Returns:
        out_df: pandas df with row for each image and the following columns:
            img_id, labels, dataset, mean probs (list of len(n_classes)), mean uncertainty (list of len(n_classes))
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if balanced == True:
        path_suf = '_balanced'
    else:
        path_suf = ''

    model = ResNet().to(device) # runs on single GPU
    # Get model
    resnet_path = os.path.join(modeldir, f"ResNet_{M}{path_suf}.pth") # uses first ResNet model by default
    state = torch.load(resnet_path, map_location=device)
    # Remove 'module.' if present from DataParallel wrapper during training
    state = {k.replace("module.", ""): v for k, v in state.items()}
    # Load from save point
    model.load_state_dict(state)
        
    # init output holders
    probs_all = []
    uncer_all = []
    labels_all = []
    img_ids_all = []
    datasets_all = []
    
    model.eval()
    with torch.no_grad():
        for name, data in data_dict.items():
            logger.info(f'\tOutputting {name}')
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
            # init per-batch holders
            batch_probs, batch_uncer, batch_labels = [], [], []
            # get outputs
            for dat, lbs, ids in data_loader:
                dat = dat.to(device, non_blocking=True) # push to device
                
                if isinstance(model, torch.nn.DataParallel):
                    probs, uncer = model.module.forward_mc(dat, repeat=5) # Runs 5 forward passes with dropout by default
                else:
                    probs, uncer = model.forward_mc(dat, repeat=5)
                
                # update batch holders
                batch_probs.append(probs.cpu())
                batch_uncer.append(uncer.cpu())
                batch_labels.append(lbs)
                img_ids_all.extend(ids)
                datasets_all.extend([name] * len(ids))
                
            # concat holders once per dataset
            probs_all.append(torch.cat(batch_probs))
            uncer_all.append(torch.cat(batch_uncer))
            labels_all.append(torch.cat(batch_labels))
        
    # final concat after all datasets are evaluated
    img_ids_all = np.array(img_ids_all)
    datasets_all = np.array(datasets_all)
    labels_all = torch.cat(labels_all).numpy()
    probs_all = torch.cat(probs_all).numpy()
    uncer_all = torch.cat(uncer_all).numpy()
    
    out_df = pd.DataFrame({
        'img_id': img_ids_all,
        'data': datasets_all,
        'labels': labels_all,
        'mean_probs': list(probs_all),
        'uncertainty': list(uncer_all)
    })
    
    return out_df

def get_embedding(model, data_dict, dim, batch_size=128, softmax=False):
    # init loss functions
    rloss_fxn = nn.MSELoss(reduction='none') # mean is applied in eval loop
    if softmax == True:
        sm = nn.Softmax(dim=1) # row-wise softmax
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # init defaultdict
    all_outputs = defaultdict(list)
    # loop datasets
    model = model.eval()
    for name, data in data_dict.items():
        logger.info(f'\tOutputting {name}')
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        # init per-batch holders
        batch_latents, batch_rloss, batch_logits, batch_labels, batch_ids = [], [], [], [], []
        with torch.inference_mode():
            # get outputs
            for dat, lbls, ids in data_loader:
                dat, lbls, = dat.to(device, non_blocking=True), lbls.to(device, non_blocking=True) # push to device
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
    logger.info('Making dataframe')
    latent_cols = ['latent'+str(i) for i in range(1, dim+1)] 
    conf_cols = ['conf'+str(i) for i in range(get_model_component(model, 'num_classes'))]
    col_names = ['img_id', 'data', 'labels'] + latent_cols + ['rloss'] + conf_cols
    
    meta_df = pd.DataFrame({
        'img_id': ids_array,
        'data': data_array,
        'labels': labels_array,
        'rloss': rloss_array,
        'dim': dim
    })
    
    # add confidence scores to new df
    conf_df = pd.DataFrame(logits_array, columns=conf_cols)
    
    # add latent embeddings to new df
    latent_df = pd.DataFrame(latent_array, columns=latent_cols)

    # concat dfs
    out_df = pd.concat([meta_df, conf_df, latent_df], axis=1)
    
    return out_df

def get_max_conf(out_df, n_classes):
    conf_cols = ['conf'+str(i) for i in range(n_classes)]
    task_data = out_df[conf_cols]
    out_df['task'] = task_data.max(axis=1) # maximum confidence value for rows
    return out_df

def get_latent_distance(latent_df, datasets, dim, k=25, reference='train', metric='manhattan'):
    dist_arr = []
    latent_cols = ['latent'+str(i) for i in range(1, dim+1)]
    # get training latent space
    ref_latent = latent_df.loc[latent_df['data'] == reference][latent_cols].to_numpy()
    # fit BallTree to train latent space, use L1 distance metric as default
    tree = BallTree(ref_latent, metric=metric)
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
        distances = np.mean(distances, axis=1)
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