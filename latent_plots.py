import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
import string
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, root_mean_squared_error, auc 
from sklearn.preprocessing import LabelBinarizer
from sklearn.inspection import permutation_importance
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from data_setup import DataBuilder

def examples_grid_mnist(builder, datasets, save=False, save_path=None):
    # Plot transforms in grid
    train_data = datasets['train'].data
    train_labels = datasets['train'].targets
    for i in range(10):
        for idx, lbl in enumerate(train_labels):
            if int(lbl) == i:
                image = train_data[idx].unsqueeze(0) # reshapes image to (C, H, W)
                fig, axs = plt.subplots(3,3, squeeze=True)
                fig.tight_layout()
                axs[0, 0].imshow(builder.transform(image).reshape(28,28), cmap="gray")
                axs[0, 0].set(title='Reference')
                axs[0, 1].imshow(builder.invert_transform(image).reshape(28,28), cmap="gray")
                axs[0, 1].set(title='Pixel Inversion')
                axs[0, 2].imshow(builder.vert_transform(image).reshape(28,28), cmap="gray")
                axs[0, 2].set(title='Vertical Flip')
                axs[1, 0].imshow(builder.horiz_transform(image).reshape(28,28), cmap="gray")
                axs[1, 0].set(title='Horiztonal Flip')
                axs[1, 1].imshow(builder.blur_transform(image).reshape(28,28), cmap="gray")
                axs[1, 1].set(title='Image Blur')
                axs[1, 2].imshow(builder.elastic50_transform(image).reshape(28,28), cmap="gray")
                axs[1, 2].set(title='Elastic Low')
                axs[2, 0].imshow(builder.elastic200_transform(image).reshape(28,28), cmap="gray")
                axs[2, 0].set(title='Elastic High')
                axs[2, 1].imshow(builder.noise20_transform(image).reshape(28,28), cmap="gray")
                axs[2, 1].set(title='Noise Low')
                axs[2, 2].imshow(builder.noise80_transform(image).reshape(28,28), cmap="gray")
                axs[2, 2].set(title='Noise High')
                # remove the x and y ticks
                plt.setp(axs, xticks=[], yticks=[])
                if save == True:
                    join_path = os.path.join(save_path, f'transform_examples_{i}.png')
                    plt.savefig(join_path, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                break
            else:
                continue

def denormalize_image(image, mean, std):
    """
    Denormalizes an image by reversing the normalization using the provided mean and std.
    Assumes image is in the shape (C, H, W) and in range [-1, 1].
    """
    for c in range(3):  # for each channel (RGB)
        image[c] = image[c] * std[c] + mean[c]  # Apply inverse normalization
    return image
    
def examples_grid_cifar(builder, datasets, save=False, save_path=None):
    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.247, 0.243, 0.261]
    # Plot transforms in grid
    train_data = datasets['train'].data
    train_labels = datasets['train'].targets
    for i in range(10):
        for idx, lbl in enumerate(train_labels):
            if int(lbl) == i:
                image = train_data[idx]
                # Ensure image is a PyTorch tensor of shape (C, H, W)
                image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
                
                fig, axs = plt.subplots(3, 4, squeeze=False, figsize=(10, 8))
                fig.subplots_adjust(hspace=0.2, wspace=0.2)
                #fig.tight_layout()

                # Apply the transform, permute, and denormalize
                img_ref = builder.transform(image)
                img_ref = denormalize_image(img_ref.clone(), cifar_mean, cifar_std).clamp(0, 1)
                axs[0, 0].imshow(img_ref.permute(1, 2, 0).numpy())
                axs[0, 0].set(title='Reference')

                img_invert = builder.invert_transform(image)
                img_invert = denormalize_image(img_invert.clone(), cifar_mean, cifar_std).clamp(0, 1)
                axs[0, 1].imshow(img_invert.permute(1, 2, 0).numpy())
                axs[0, 1].set(title='Pixel Inversion')

                img_vert = builder.vert_transform(image)
                img_vert = denormalize_image(img_vert.clone(), cifar_mean, cifar_std).clamp(0, 1)
                axs[0, 2].imshow(img_vert.permute(1, 2, 0).numpy())
                axs[0, 2].set(title='Vertical Flip')

                img_horiz = builder.horiz_transform(image)
                img_horiz = denormalize_image(img_horiz.clone(), cifar_mean, cifar_std).clamp(0, 1)
                axs[0, 3].imshow(img_horiz.permute(1, 2, 0).numpy())
                axs[0, 3].set(title='Horizontal Flip')

                img_blur = builder.blur_transform(image)
                img_blur = denormalize_image(img_blur.clone(), cifar_mean, cifar_std).clamp(0, 1)
                axs[1, 0].imshow(img_blur.permute(1, 2, 0).numpy())
                axs[1, 0].set(title='Image Blur')

                img_elastic50 = builder.elastic50_transform(image)
                img_elastic50 = denormalize_image(img_elastic50.clone(), cifar_mean, cifar_std).clamp(0, 1)
                axs[1, 1].imshow(img_elastic50.permute(1, 2, 0).numpy())
                axs[1, 1].set(title='Elastic Low')

                img_elastic200 = builder.elastic200_transform(image)
                img_elastic200 = denormalize_image(img_elastic200.clone(), cifar_mean, cifar_std).clamp(0, 1)
                axs[1, 2].imshow(img_elastic200.permute(1, 2, 0).numpy())
                axs[1, 2].set(title='Elastic High')

                img_noise20 = builder.noise20_transform(image)
                img_noise20 = denormalize_image(img_noise20.clone(), cifar_mean, cifar_std).clamp(0, 1)
                axs[1, 3].imshow(img_noise20.permute(1, 2, 0).numpy())
                axs[1, 3].set(title='Noise Low')

                img_noise80 = builder.noise80_transform(image)
                img_noise80 = denormalize_image(img_noise80.clone(), cifar_mean, cifar_std).clamp(0, 1)
                axs[2, 0].imshow(img_noise80.permute(1, 2, 0).numpy())
                axs[2, 0].set(title='Noise High')

                img_posterize = builder.posterize_transform(image)
                img_posterize = denormalize_image(img_posterize.clone(), cifar_mean, cifar_std).clamp(0, 1)
                axs[2, 1].imshow(img_posterize.permute(1, 2, 0).numpy())
                axs[2, 1].set(title='Posterize')

                img_solarize = builder.solarize_transform(image)
                img_solarize = denormalize_image(img_solarize.clone(), cifar_mean, cifar_std).clamp(0, 1)
                axs[2, 2].imshow(img_solarize.permute(1, 2, 0).numpy())
                axs[2, 2].set(title='Solarize')

                axs[2, 3].axis('off')
                
                # remove the x and y ticks
                plt.setp(axs, xticks=[], yticks=[])
                if save == True:
                    join_path = os.path.join(save_path, f'transform_examples_{i}.png')
                    plt.savefig(join_path, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                break
            else:
                continue

def original_vs_reconstruction_mnist(model, datasets, recon_loss=False, save=False, save_path=None):
    # init reconstruction loss function
    rloss_fxn = torch.nn.MSELoss()
    # show training original image vs. reconstruction
    test_data = datasets['test'].data
    test_labels = datasets['test'].targets
    builder = DataBuilder()
    model.eval()
    for i in range(10):
        for idx, lbl in enumerate(test_labels):
            if int(lbl) == i:
                # get original img and plot
                img = builder.transform(test_data[idx])
                fig, ax = plt.subplots(figsize=(2,2))
                ax.imshow(img.view(28, 28), cmap='gray')
                # remove the x and y ticks
                plt.setp(ax, xticks=[], yticks=[])
                if save == True:
                    join_path_orig = os.path.join(save_path, f'original_{i}.png')
                    plt.savefig(join_path_orig, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # get reconstructed img and loss
                img = img.to(torch.float32)
                with torch.no_grad():
                    img = img.unsqueeze(0) # adds channel to img
                    _, recon, _ = model(img)
                if recon_loss:
                    r_loss = rloss_fxn(recon, img)
                    r_loss = round(float(r_loss), 3)
                # plot reconstruction
                fig, ax = plt.subplots(figsize=(2,2))
                ax.imshow(recon.view(28, 28), cmap='gray')
                if recon_loss:
                    ax.set_title(f'Loss = {r_loss}', fontsize=18, pad=10)
                # remove the x and y ticks
                plt.setp(ax, xticks=[], yticks=[])
                if save == True:
                    join_path_recon = os.path.join(save_path, f'reconstructed_{i}.png')
                    plt.savefig(join_path_recon, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                break
            else:
                continue

def confusion_matrix_from_df(df, save=False, save_path=None):
    labels = df.loc[df['data'] == 'test']['labels'].to_numpy()
    sub_df = df.loc[df['data'] == 'test'][['conf'+str(i) for i in range(10)]].to_numpy()
    preds = sub_df.argmax(axis=1)
    f1 = f1_score(labels, preds, average='micro')
    # make confusion matrix and save
    conf_mat = ConfusionMatrixDisplay.from_predictions(preds, labels)
    conf_mat.ax_.set_title(f'F1 Score = {f1}', pad=10, fontsize=14)
    if save == True:
        joined_path = os.path.join(save_path, f'test_confusionmatrix.png')
        plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
    plt.close()

def overlapped_latent_plot(latent_df, save=False, save_path=None):
    # get latent embeddings for train and test sets
    train_1 = latent_df.loc[latent_df['data'] == 'train'][['latent1']]
    train_2 = latent_df.loc[latent_df['data'] == 'train'][['latent2']]
    test_1 = latent_df.loc[latent_df['data'] == 'test'][['latent1']]
    test_2 = latent_df.loc[latent_df['data'] == 'test'][['latent2']]
    # get train and test labels
    train_labels = latent_df.loc[latent_df['data'] == 'train'].labels
    test_labels = latent_df.loc[latent_df['data'] == 'test'].labels
    # init figure
    fig, ax = plt.subplots(figsize=(8, 6))
    # plot train data
    colors1 = ListedColormap(plt.get_cmap('tab20').colors[1::2])
    scatter1 = ax.scatter(train_1, train_2, s=5, c=train_labels, cmap=colors1, alpha=0.75)
    # plot test data
    colors2 = ListedColormap(plt.get_cmap('tab20').colors[::2])
    scatter2 = ax.scatter(test_1, test_2, s=5, c=test_labels, cmap=colors2)
    # make train and test legends
    legend_handles = []
    legend_labels = []
    # train
    for i, color in enumerate(colors1.colors):
        legend_handles.append(
            Line2D(
                [0], [0], 
                linestyle='None', marker='o', markersize=10, color='w', markerfacecolor=color, label=str(i)
            )
        )
        legend_labels.append(str(''))
    # test
    for i, color in enumerate(colors2.colors):
        legend_handles.append(
            Line2D(
                [0], [0], 
                linestyle='None', marker='o', markersize=10, color='w', markerfacecolor=color, label=str(i)
            )
        )
        legend_labels.append(str(i))
    ax.legend(handles=legend_handles, labels=legend_labels, ncol=2, columnspacing=0.25, title='Train  Test')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    if save == True:
        joined_path = os.path.join(save_path, f'overlapped_latent_plot.png')
        plt.savefig(joined_path, format='png', dpi=300)
    plt.close()

def trainonly_latent_plot(latent_df, labels_dict, save=False, save_path=None):
    # get latent embeddings for train set
    train_1 = latent_df.loc[latent_df['data'] == 'train'][['latent1']]
    train_2 = latent_df.loc[latent_df['data'] == 'train'][['latent2']]
    # get train labels
    train_labels = latent_df.loc[latent_df['data'] == 'train'].labels
    # init figure
    fig, ax = plt.subplots(figsize=(8, 6))
    # plot train data
    colors1 = ListedColormap(plt.get_cmap('tab20').colors[1::2])
    scatter1 = ax.scatter(train_1, train_2, s=5, c=train_labels, cmap=colors1, alpha=0.9)
    # make legend
    legend_handles = []
    legend_labels = []
    # train
    for i, color in enumerate(colors1.colors):
        legend_handles.append(
            Line2D(
                [0], [0], 
                linestyle='None', marker='o', markersize=10, color='w', markerfacecolor=color, label=str(i)
            )
        )
        legend_labels.append(labels_dict[i])
    ax.legend(handles=legend_handles, labels=legend_labels, ncol=2, columnspacing=0.25, title='CIFAR-10 Class')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    if save == True:
        joined_path = os.path.join(save_path, f'trainonly_latent_plot.png')
        plt.savefig(joined_path, format='png', dpi=300)
    plt.close()

def latent_density_maps(latent_df, probs_df, save=False, save_path=None):
    meas_dict = {
        'count':'Count',
        'rloss':'Avg. Reconstruction Error',
        'task':'Avg. Classification Confidence',
        'gmean_metric':'Avg. SAGE Score'
    }
    sub_latent = latent_df.loc[latent_df['data'] == 'train'] 
    # plot binned density
    x_bin = pd.cut(sub_latent['latent1'], bins=100, labels=list(range(100)))
    y_bin = pd.cut(sub_latent['latent2'], bins=100, labels=list(range(100)))
    sub_latent['x_bin'] = x_bin
    sub_latent['y_bin'] = y_bin
    # density binned heatmap
    sub_latent['tally'] = list(range(1, len(sub_latent)+1))
    sub_latent['gmean_metric'] = probs_df.loc[probs_df['data'] == 'train'].gmean_metric
    for meas in ['rloss', 'task', 'count', 'gmean_metric']:
        if meas == 'count':
            density_df = sub_latent.pivot_table(index='y_bin', columns='x_bin', values='tally', aggfunc='sum').replace(0, np.nan)
        else:
            density_df = sub_latent.pivot_table(index='y_bin', columns='x_bin', values=meas, aggfunc='mean').replace(0, np.nan)
        mask = density_df.isnull()
        
        fig, ax = plt.subplots()
        if meas == 'rloss':
            rl_arr = sub_latent[meas].to_numpy()
            vmax = np.percentile(rl_arr, 95)
        else:
            vmax=None
        # make binned heatmap
        hm = sns.heatmap(
            data=density_df.sort_index(ascending=False), ax=ax,
            linewidths=0, clip_on=False, mask=mask.sort_index(ascending=False),
            cmap='viridis', cbar_kws={"shrink": .75, "label":meas_dict[meas]}, vmax=vmax)
        # Drawing the frame 
        for _, spine in hm.spines.items(): 
            spine.set_visible(True) 
            spine.set_linewidth(1)
            spine.set_color('lightgrey')
        plt.xlabel('Latent Dimension 1', labelpad=15)
        plt.ylabel('Latent Dimension 2', labelpad=15)
        ax.set_yticklabels('')
        ax.get_yaxis().set_ticks([])
        ax.set_xticklabels('')
        ax.get_xaxis().set_ticks([])
        if save == True:
            joined_path = os.path.join(save_path, f'latent_{meas}_binned.png')
            plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
        plt.close()

def mnist_custom_embedding_plot(latent_df, save=False, save_path=None):
    '''
    '''
    name_dict = {
        'test':'Original',
        'elastic50':'Elastic Low',
        'noise20':'Noise Low',
        'blur':'Blur',
        'invert':'Pixel Inversion',
        'elastic200':'Elastic High',
        'noise80':'Noise High',
        'vertical':'Vertical Flip',
        'horizontal':'Horizontal Flip',
    }
    # test index == 4 for same '5' image used in original/reconstruction plots
    img_idx = 4
    builder = DataBuilder()
    datasets = builder.build_mnist()

    #colors = plt.get_cmap('tab10').colors
    letters = list(string.ascii_uppercase)
    
    # init figure and listed colormap
    fig, ax = plt.subplots(figsize=(8, 6))

    # plot sample of training data
    train_df = latent_df.loc[latent_df['data'] == 'train'].sample(40000, random_state=33)
    train_latent = train_df[['latent1', 'latent2']].to_numpy()
    ax.scatter(train_latent[:, 0], train_latent[:, 1], s=20, alpha=0.3, color='darkgray')
    
    order = []
    for i, name in enumerate(name_dict.keys()):
        # plot single transformed point for '5'
        name_df = latent_df.loc[latent_df['data'] == name].reset_index(drop=True)
        point_df = name_df.iloc[img_idx]
        point_coords = point_df[['latent1', 'latent2']].to_numpy()
        # plot dot at image location
        x = point_coords[0]
        y = point_coords[1]
        order.append((name, point_coords)) # retains x, y values where to plot points

    x0, y0 = -103, 65 # set standard x, y values for image plots
    width, height = 18, 18 # set image dimensions
    for i, group in enumerate(order): # list of tuples with (name, [x, y])
        if i == 0: # plot original '5' image
            ax.scatter(group[1][0], group[1][1], s=90, color='sienna', marker='*', label=f'{letters[i]} = {name_dict[group[0]]}')
            ax.text(group[1][0]+2, group[1][1], str(letters[i]), {'fontsize':14, 'fontweight':'bold'})
        else:
            ax.scatter(group[1][0], group[1][1], s=20, color='black', label=f'{letters[i]} = {name_dict[group[0]]}')
            ax.text(group[1][0]+2, group[1][1], str(letters[i]), {'fontsize':14, 'fontweight':'bold'})
        # get image and plot above dot
        img = datasets[group[0]][4][0]
        # Draw image
        new_ax = ax.inset_axes([x0, y0, width, height], transform=ax.transData) # create new inset axes in data coordinates
        new_ax.imshow(img.reshape(28,28), cmap='gray')
        new_ax.set_xticks([])
        new_ax.set_yticks([])
        new_ax.set_title(letters[i], fontsize=11, loc='center')
        x0 += 18 # increment image position
    plt.xlabel('Latent Dimension 1', fontsize=13, labelpad=10)
    plt.ylabel('Latent Dimension 2', fontsize=13, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim([-105, 60])
    if save == True:
        joined_path = os.path.join(save_path, 'transform_embedding_examples.png')
        plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def measures_correlation_heatmap(latent_df, save=False, save_path=None):
    plt.figure(figsize = (6, 6))
    heatmap = sns.heatmap(
        latent_df[['dist', 'rloss', 'task']].corr(),
        cmap='cividis',
        vmin = -1, 
        vmax = 1, 
        annot = True, 
        square=True,
        xticklabels = ['Avg. kNN\nDistance', 'Reconstruction\nError', 'Classifier\nConfidence'],
        yticklabels = ['Avg. kNN\nDistance', 'Reconstruction\nError', 'Classifier\nConfidence'],
        annot_kws={"size":11},
        cbar_kws={"shrink": .75, "label":'Correlation Coefficient', "ticks":[1,0.5,0,-0.5,-1]}
    )
    heatmap.tick_params(axis='both', which='major', labelsize=11, pad=10)
    if save == True:
        joined_path = os.path.join(save_path, 'output_measures_heatmap.png')
        plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def abalone_correlation_heatmap(latent_df, save=False, save_path=None):
    plt.figure(figsize = (6, 6))
    heatmap = sns.heatmap(
        latent_df[['dist', 'rloss', 'task']].corr(),
        cmap='cividis',
        vmin = -1, 
        vmax = 1, 
        annot = True, 
        square=True,
        xticklabels = ['Avg. kNN\nDistance', 'Reconstruction\nError', 'Regression\nError'],
        yticklabels = ['Avg. kNN\nDistance', 'Reconstruction\nError', 'Regression\nError'],
        annot_kws={"size":11},
        cbar_kws={"shrink": .75, "label":'Correlation Coefficient', "ticks":[1,0.5,0,-0.5,-1]}
    )
    heatmap.tick_params(axis='both', which='major', labelsize=11, pad=10)
    if save == True:
        joined_path = os.path.join(save_path, 'output_measures_heatmap.png')
        plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
        
def mnist_output_boxplots(latent_df, name_dict, save=False, save_path=None):
    meas_dict = {
        'dist':'kNN Distance to Train',
        'rloss':'Reconstruction Error',
        'task':'Classification Confidence',
    }
    order_list = [
        'test', 
        'elastic50', 
        'noise20', 
        'blur', 
        'invert', 
        'elastic200', 
        'noise80', 
        'vertical', 
        'horizontal'
    ]
    assert len(latent_df.dim.unique()) == 1
    dim = latent_df.dim.unique()[0]
    # make boxplots
    plot_df = latent_df.loc[latent_df['data'].isin(name_dict.keys())]
    plot_df['data'] = plot_df['data'].replace(name_dict)
    for meas in ['dist', 'rloss', 'task']:
        fig, ax = plt.subplots()
        sns.boxplot(
            data=plot_df, 
            x='data', 
            y=meas, 
            color='lightsteelblue', 
            showfliers=False, 
            ax=ax, 
            boxprops={'edgecolor': 'black'},
            whiskerprops={'color':'black'}, 
            capprops={'color':'black'}, 
            medianprops={'color':'black'}
        )
        #ax.tick_params(axis='x', labelrotation=45, labelsize=11)
        ax.tick_params(axis='x', labelsize=10)
        ax.set_ylabel(meas_dict[meas], labelpad=10, fontsize=11)
        ax.set_xlabel('')
        plt.yticks(fontsize=11)
        if save == True:
            joined_path = os.path.join(save_path, f'mnist_convsae{dim}D_{meas}boxplots.png')
            plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # test distance shift boxplot
    sub_df = latent_df.loc[latent_df['data'].isin(list(name_dict.keys())[1:])][['data', 'dist']] # skip train
    test_dist = sub_df.loc[sub_df['data'] == 'test']['dist'].to_numpy()
    test_dist = np.tile(test_dist, int(len(sub_df)/len(test_dist))) # repeat test distances to span df length
    sub_df['dist'] -= test_dist
    sub_df = sub_df.sort_values('data', key=lambda s: s.apply(order_list.index), ignore_index=True)
    sub_df['data'] = sub_df['data'].replace(name_dict)
    fig, ax = plt.subplots()
    colors = sns.color_palette("colorblind").as_hex()
    sns.boxplot(
        data=sub_df, 
        x='data', 
        y='dist', 
        color=colors[2], 
        showfliers=False, 
        ax=ax, 
        boxprops={'edgecolor':'black'},
        whiskerprops={'color':'black'}, 
        capprops={'color':'black'}, 
        medianprops={'color':'black'}
    )
    ax.tick_params(axis='x', labelsize=10.5)
    ax.set_ylabel('kNN Distance Shift', labelpad=10, fontsize=11)
    ax.set_xlabel('')
    plt.yticks(fontsize=11)
    if save == True:
        joined_path = os.path.join(save_path, f'mnist_convsae{dim}D_testdistanceshift.png')
        plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
    plt.close()

def cifar_output_boxplots(latent_df, name_dict, save=False, save_path=None):
    meas_dict = {
        'dist':'kNN Distance to Train',
        'rloss':'Reconstruction Error',
        'task':'Classification Confidence',
    }
    assert len(latent_df.dim.unique()) == 1
    dim = latent_df.dim.unique()[0]
    # make boxplots
    plot_df = latent_df.loc[latent_df['data'].isin(name_dict.keys())]
    plot_df['data'] = plot_df['data'].replace(name_dict)
    for meas in ['dist', 'rloss', 'task']:
        fig, ax = plt.subplots()
        sns.boxplot(
            data=plot_df, 
            x='data', 
            y=meas, 
            color='lightsteelblue', 
            showfliers=False, 
            ax=ax, 
            boxprops={'edgecolor': 'black'},
            whiskerprops={'color':'black'}, 
            capprops={'color':'black'}, 
            medianprops={'color':'black'}
        )
        #ax.tick_params(axis='x', labelrotation=45, labelsize=11)
        ax.tick_params(axis='x', labelsize=9)
        ax.set_ylabel(meas_dict[meas], labelpad=10, fontsize=11)
        ax.set_xlabel('')
        plt.yticks(fontsize=11)
        if save == True:
            joined_path = os.path.join(save_path, f'cifar_resnetsae{dim}D_{meas}_boxplots.png')
            plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
        plt.close()
    
def train_ne_curves(latent_df, save=False, save_path=None):
    '''
    '''
    dist = np.log(latent_df.loc[latent_df.data == 'train'].dist.to_numpy())
    rloss = latent_df.loc[latent_df.data == 'train'].rloss.to_numpy()
    task = -np.log(latent_df.loc[latent_df.data == 'train'].task.to_numpy())
    colors = sns.color_palette("colorblind").as_hex()
    
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, layout='constrained', figsize=(5, 5), gridspec_kw={'hspace':0.1})
    for i, m in enumerate(['dist', 'rloss', 'task']):
        array = eval(m)
        array = np.sort(array)
        probs = [1 - (j+1)/len(array) for j, _ in enumerate(array)]
        ax = eval('ax'+str(i))
        ax.plot(array, probs, color=colors[0], lw=2)
        ax.fill_between(array, probs, color=colors[0], alpha=0.5)
        if m == 'task':
            ax.set_xlim([-0.0025, 0.055])
        if m == 'rloss':
            ax.set_xlim([-0.0025, 0.17])
    
    ax0.set_xlabel('log Avg. kNN Distance', fontsize=12, labelpad=5)
    ax0.set_ylabel('')
    ax0.tick_params(axis='both', which='major', labelsize=11)
    
    ax1.set_xlabel('Reconstruction Error', fontsize=12, labelpad=5)
    ax1.set_ylabel('Exceedance Probability', fontsize=12, labelpad=10)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    
    ax2.set_xlabel('-log Classifier Confidence', fontsize=12, labelpad=5)
    ax2.set_ylabel('')
    ax2.tick_params(axis='both', which='major', labelsize=11)

    if save == True:
        joined_path = os.path.join(save_path, 'train_output_ne_curves.png')
        plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def simple_quantile_plots(latent_df, save=False, save_path=None):
    '''
    '''
    colors = sns.color_palette("colorblind").as_hex()
    for i, name in enumerate(['train', 'test', 'transform']):
        if name == 'transform':
            out = latent_df.loc[(latent_df.data != 'train') & (latent_df.data != 'test')]
        else:
            out = latent_df.loc[latent_df.data == name]
        # set up image variables
        #y_lims = [[-1.5, 5.5], [-0.05, 0.8], [-0.1, 1.9]]
        #for lim, meas in zip(y_lims, ['dist', 'rloss', 'task']):
        for meas in ['dist', 'rloss', 'task']:
            meas_samp = out[['data', meas]]
            counts = list(range(1, len(meas_samp)+1))
            # set other variables        
            fig, ax = plt.subplots(figsize=(7, 4))
            scores = meas_samp[meas].to_numpy()
            if meas == 'dist':
                ylabel = 'log Avg. kNN Distance'
                scores = np.log(scores)
            elif meas == 'task':
                ylabel = '-log Classifier Confidence'
                scores = -np.log(scores)
            else:
                ylabel = 'Reconstruction Error'
            # sort scores
            scores = np.sort(scores)
            ax.plot(counts, scores, color=colors[i], lw=3)
            ax.grid(visible=True, which='both')
            ax.set_xlabel('Image Quantile', fontsize=13, labelpad=7.5)
            #ax.set_ylim(lim)
            ax.set_ylabel(ylabel, fontsize=13, labelpad=7.5)
            ax.tick_params(axis='both', which='major', labelsize=11)
            if save == True:
                joined_path = os.path.join(save_path + f'{name}_{meas}_quantiles.png')
                plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()

def probability_violinplots(probs_df, name_dict, save=False, save_path=None):
    fig, ax = plt.subplots()
    # original hist
    train_array = probs_df.loc[probs_df.data == 'train']['gmean_metric'].to_numpy()
    test_array = probs_df.loc[probs_df.data == 'test']['gmean_metric'].to_numpy()
    #transformed hist
    trans_array = probs_df.loc[(probs_df.data != 'train') & (probs_df.data != 'test')]['gmean_metric'].to_numpy()
    vio_df = pd.DataFrame({
        'Score':np.concatenate((train_array, test_array, trans_array), axis=None),
        'Data':np.concatenate(
            (
                np.array(['Train']*len(train_array)),
                np.array(['Test']*len(test_array)),
                np.array(['Transformed']*len(trans_array))
            ),
            axis=None
        )
    })
    sns.violinplot(vio_df, x='Data', y='Score', alpha=0.7, hue='Data', palette='colorblind')
    ax.set_ylabel('SAGE Score', fontsize=13, labelpad=15)
    ax.set_xlabel('Dataset', fontsize=13, labelpad=15)
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='y', which='major', labelsize=11)
    
    if save == True:
        joined_path = os.path.join(save_path + f'all_prob_violin.png')
        plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    # make violinplot of trans SAGE scores by transform
    fig, ax = plt.subplots()
    colors = sns.color_palette("colorblind").as_hex()
    trans_names = probs_df.loc[(probs_df.data != 'train') & (probs_df.data != 'test')]['data'].replace(name_dict).to_numpy()
    trans_vio_df = pd.DataFrame({
        'Score':trans_array,
        'Data':trans_names
    })
    sns.violinplot(trans_vio_df, x='Data', y='Score', hue='Data', alpha=0.7, color=colors[2])
    ax.set_ylabel('SAGE Score', fontsize=13, labelpad=15)
    ax.tick_params(axis='x', which='major', labelsize=10.5)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.set_xlabel('')
    if save == True:
        joined_path = os.path.join(save_path + f'transforms_prob_violin.png')
        plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def trans_probability_quantiles(probs_df, datasets, save=False, save_path=None):
    '''
    '''
    colors = sns.color_palette("colorblind").as_hex()
    # plot all transformed images
    trans_df = probs_df.loc[(probs_df.data != 'train') & (probs_df.data != 'test')]
    trans_df = trans_df[['data', 'dist_prob', 'rloss_prob', 'task_prob', 'gmean_metric']].sort_values('gmean_metric')
    counts = list(range(1, len(trans_df)+1))
    # set other variables
    row_counts = probs_df.data.value_counts()
    target_values = np.append(np.linspace(0, 63999, 11, dtype=int)[:-1], 63998) # prevents IndexError
    width, length = 5270, 5270
    
    fig, ax = plt.subplots(figsize=(7, 4))
    # plot score vs. sorted image curves
    scores = trans_df.gmean_metric.to_numpy() # scores are pre-sorted
    ax.plot(counts, scores, color=colors[2], lw=3)
    
    # plot example images above the hist
    for j, v in enumerate(target_values):
        x0 = v - 2447
        ymin = scores[v]
        ax.axvline(x=v, ymin=ymin+0.02, ymax=1, color='black', lw=1, ls=':') # draw vertical line
        ax.scatter(v, ymin, color=colors[2], marker='o', s=30)
        text = str(round(ymin, 2))
        if 0 < ymin < 0.01:
            ax.text(x=v-(550*5), y=1.05, s=f"{ymin:.0e}", fontdict={'fontsize':11})
        else:
            if len(re.search('(?<=\.).+', str(text)).group(0)) == 1:
                ax.text(x=v-(395*5), y=1.05, s=text+'0', fontdict={'fontsize':11})
            else:
                ax.text(x=v-(395*5), y=1.05, s=text, fontdict={'fontsize':11})
        for n in range(2):
            row = trans_df.iloc[v+n]
            ind = row_counts.index.to_list().index(row.data) # get index of dataset w/in value counts
            floor = sum([row_counts.iloc[i] for i in range(ind)]) # get the number of preceding images in prob_df 
            img_loc = row.name - floor # gets index of original test image to transform
            img = datasets[row.data][img_loc][0] # pulls the transformed image
            y0 = 1.1 + (n * .14)
            new_ax = ax.inset_axes([x0, y0, 5270, .15], transform=ax.transData) # create new inset axes in data coordinates
            new_ax.imshow(img.reshape(28,28), cmap='gray') # plot image
            new_ax.set_xticks([])
            new_ax.set_yticks([])
    ax.set_xlabel('Image Quantile', fontsize=13, labelpad=7.5)
    ax.set_ylim([0, 1])
    #ax.set_xlim([-0.05, 1.0])
    ax.set_ylabel('SAGE Score', fontsize=13, labelpad=7.5)
    ax.tick_params(axis='both', which='major', labelsize=11)
    if save == True:
        joined_path = os.path.join(save_path + f'trans_prob_quantiles.png')
        plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def trans_measure_quantiles(probs_df, datasets, save=False, save_path=None):
    meas_dict = {
    'dist_prob':'Distance Probability',
    'rloss_prob':'Reconstruction Probability',
    'task_prob':'Classifier Probability'
    }
    trans_df = probs_df.loc[(probs_df.data != 'train') & (probs_df.data != 'test')]
    colors = sns.color_palette("colorblind").as_hex()
    for meas in ['dist_prob', 'rloss_prob', 'task_prob']:
        meas_samp = trans_df[['data', meas]].sort_values(meas)
        counts = list(range(1, len(meas_samp)+1))
        # set other variables
        row_counts = probs_df.data.value_counts()
        target_values = np.append(np.linspace(0, 63999, 11, dtype=int)[:-1], 63998) # prevents IndexError
        width, length = 5270, 5270
        
        fig, ax = plt.subplots(figsize=(7, 4))
        # plot score vs. sorted image curves
        scores = meas_samp[meas].to_numpy() # scores are pre-sorted
        ax.plot(counts, scores, color=colors[2], lw=3)
        
        # plot example images above the hist
        for j, v in enumerate(target_values):
            x0 = v - 2447
            ymin = scores[v]
            ax.axvline(x=v, ymin=ymin+0.02, ymax=1, color='black', lw=1, ls=':') # draw vertical line
            ax.scatter(v, ymin, color=colors[2], marker='o', s=30)
            text = str(round(ymin, 2))
            if 0 < ymin < 0.01:
                ax.text(x=v-(550*5), y=1.05, s=f"{ymin:.0e}", fontdict={'fontsize':11})
            else:
                if len(re.search('(?<=\.).+', str(text)).group(0)) == 1:
                    ax.text(x=v-(395*5), y=1.05, s=text+'0', fontdict={'fontsize':11})
                else:
                    ax.text(x=v-(395*5), y=1.05, s=text, fontdict={'fontsize':11})
            for n in range(2):
                row = meas_samp.iloc[v+n]
                ind = row_counts.index.to_list().index(row.data) # get index of dataset w/in value counts
                floor = sum([row_counts.iloc[i] for i in range(ind)]) # get the number of preceding images in prob_df 
                img_loc = row.name - floor # gets index of original test image to transform
                img = datasets[row.data][img_loc][0] # pulls the transformed image
                y0 = 1.1 + (n * .14)
                new_ax = ax.inset_axes([x0, y0, 5270, .15], transform=ax.transData) # create new inset axes in data coordinates
                new_ax.imshow(img.reshape(28,28), cmap='gray') # plot image
                new_ax.set_xticks([])
                new_ax.set_yticks([])
        ax.set_xlabel('Image Quantile', fontsize=13, labelpad=7.5)
        ax.set_ylim([0, 1])
        ax.set_ylabel(meas_dict[meas], fontsize=13, labelpad=7.5)
        ax.tick_params(axis='both', which='major', labelsize=11)
        if save == True:
            joined_path = os.path.join(save_path + f'trans_{meas}_quantiles.png')
            plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()

def train_probability_quantiles(probs_df, datasets, save=False, save_path=None):
    # set up image variables
    train_df = probs_df.loc[probs_df.data == 'train']
    train_df = train_df[['data', 'gmean_metric']].sort_values('gmean_metric')
    counts = list(range(1, len(train_df)+1))
    row_counts = probs_df.data.value_counts()
    target_values = np.append(np.linspace(0, 59999, 11, dtype=int)[:-1], 59998)
    width, length = 5000, 5000
    colors = sns.color_palette("colorblind").as_hex()
    fig, ax = plt.subplots(figsize=(7, 4))
    # plot non-exceedance curves
    #cumulative_probs = [1 - (j+1)/len(train_df) for j, _ in enumerate(np.sort(train_df.gmean_metric.to_numpy()))]
    #sorted_probs = np.sort(train_df.gmean_metric.to_numpy())
    scores = train_df.gmean_metric.to_numpy() # scores are pre-sorted
    ax.plot(counts, scores, color=colors[0], lw=3)
    #ax.fill_between(sorted_probs, cumulative_probs, color='dimgrey', alpha=0.5)
    
    # plot example images above the hist
    for j, v in enumerate(target_values):
        x0 = v - 2500
        ymin = scores[v]
        ax.axvline(x=v, ymin=ymin+0.02, ymax=1, color='black', lw=1, ls=':') # draw vertical line
        ax.scatter(v, ymin, color=colors[0], marker='o', s=30)
        text = str(round(ymin, 2))
        if 0 < ymin < 0.01:
                ax.text(x=v-(500*5), y=1.05, s=f"{ymin:.0e}", fontdict={'fontsize':11})
        else:
            if len(re.search('(?<=\.).+', str(text)).group(0)) == 1:
                ax.text(x=v-(420*5), y=1.05, s=text+'0', fontdict={'fontsize':11})
            else:
                ax.text(x=v-(420*5), y=1.05, s=text, fontdict={'fontsize':11})
        for n in range(2):
            row = train_df.iloc[v+n]
            ind = row_counts.index.to_list().index(row.data) # get index of dataset w/in value counts
            floor = sum([row_counts.iloc[i] for i in range(ind)]) # get the number of preceding images in prob_df 
            img_loc = row.name - floor # gets index of original test image to transform
            img = datasets[row.data][img_loc][0] # pulls the transformed image
            y0 = 1.1 + (n * .145)
            new_ax = ax.inset_axes([x0, y0, 5000, .15], transform=ax.transData) # create new inset axes in data coordinates
            new_ax.imshow(img.reshape(28,28), cmap='gray') # plot image
            new_ax.set_xticks([])
            new_ax.set_yticks([])
    
    ax.set_xlabel('Image Quantile', fontsize=13, labelpad=7.5)
    ax.set_ylim([0, 1])
    #ax.set_xlim([-0.05, 1.05])
    ax.set_ylabel('SAGE Score', fontsize=13, labelpad=7.5)
    ax.tick_params(axis='both', which='major', labelsize=11)
    if save == True:
        joined_path = os.path.join(save_path + f'train_prob_quantiles.png')
        plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def train_measure_quantiles(probs_df, datasets, save=False, save_path=None):
    # set up image variables
    train_df = probs_df.loc[probs_df.data == 'train']
    meas_dict = {
        'dist_prob':'Distance Probability',
        'rloss_prob':'Reconstruction Probability',
        'task_prob':'Classifier Probability'
    }
    colors = sns.color_palette("colorblind").as_hex()
    for meas in ['dist_prob', 'rloss_prob', 'task_prob']:
        meas_samp = train_df[['data', meas]].sort_values(meas)
        counts = list(range(1, len(meas_samp)+1))
        row_counts = probs_df.data.value_counts()
        target_values = np.append(np.linspace(0, 59999, 11, dtype=int)[:-1], 59998)
        width, length = 5000, 5000
        
        fig, ax = plt.subplots(figsize=(7, 4))
        # plot non-exceedance curves
        #cumulative_probs = [1 - (j+1)/len(train_df) for j, _ in enumerate(np.sort(train_df.gmean_metric.to_numpy()))]
        #sorted_probs = np.sort(train_df.gmean_metric.to_numpy())
        scores = meas_samp[meas].to_numpy() # scores are pre-sorted
        ax.plot(counts, scores, color=colors[0], lw=3)
        #ax.fill_between(sorted_probs, cumulative_probs, color='dimgrey', alpha=0.5)
        
        # plot example images above the hist
        for j, v in enumerate(target_values):
            x0 = v - 2500
            ymin = scores[v]
            ax.axvline(x=v, ymin=ymin+0.02, ymax=1, color='black', lw=1, ls=':') # draw vertical line
            ax.scatter(v, ymin, color=colors[0], marker='o', s=30)
            text = str(round(ymin, 2))
            if 0 < ymin < 0.01:
                ax.text(x=v-(500*5), y=1.05, s=f"{ymin:.0e}", fontdict={'fontsize':11})
            else:
                if len(re.search('(?<=\.).+', str(text)).group(0)) == 1:
                    ax.text(x=v-(420*5), y=1.05, s=text+'0', fontdict={'fontsize':11})
                else:
                    ax.text(x=v-(420*5), y=1.05, s=text, fontdict={'fontsize':11})
            for n in range(2):
                row = meas_samp.iloc[v+n]
                ind = row_counts.index.to_list().index(row.data) # get index of dataset w/in value counts
                floor = sum([row_counts.iloc[i] for i in range(ind)]) # get the number of preceding images in prob_df 
                img_loc = row.name - floor # gets index of original test image to transform
                img = datasets[row.data][img_loc][0] # pulls the transformed image
                y0 = 1.1 + (n * .145)
                new_ax = ax.inset_axes([x0, y0, 5000, .15], transform=ax.transData) # create new inset axes in data coordinates
                new_ax.imshow(img.reshape(28,28), cmap='gray') # plot image
                new_ax.set_xticks([])
                new_ax.set_yticks([])
        
        ax.set_xlabel('Image Quantile', fontsize=13, labelpad=7.5)
        #ax.set_ylim([-0.05, 1.05])
        ax.set_ylim([0, 1])
        ax.set_ylabel(meas_dict[meas], fontsize=13, labelpad=7.5)
        ax.tick_params(axis='both', which='major', labelsize=11)
        if save == True:
            joined_path = os.path.join(save_path + f'train_{meas}_quantiles.png')
            plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()

def test_probability_quantiles(probs_df, datasets, save=False, save_path=None):
    # set up image variables
    test_df = probs_df.loc[probs_df.data == 'test']
    test_df = test_df[['data', 'gmean_metric']].sort_values('gmean_metric')
    counts = list(range(1, len(test_df)+1))
    row_counts = probs_df.data.value_counts()
    target_values = np.append(np.linspace(0, 7999, 11, dtype=int)[:-1], 7998)
    width, length = 666, 666
    colors = sns.color_palette("colorblind").as_hex()
    fig, ax = plt.subplots(figsize=(7, 4))
    # plot non-exceedance curves
    #cumulative_probs = [1 - (j+1)/len(train_df) for j, _ in enumerate(np.sort(train_df.gmean_metric.to_numpy()))]
    #sorted_probs = np.sort(train_df.gmean_metric.to_numpy())
    scores = test_df.gmean_metric.to_numpy() # scores are pre-sorted
    ax.plot(counts, scores, color=colors[1], lw=3)
    #ax.fill_between(sorted_probs, cumulative_probs, color='dimgrey', alpha=0.5)
    
    # plot example images above the hist
    for j, v in enumerate(target_values):
        x0 = v - 333
        ymin = scores[v]
        ax.axvline(x=v, ymin=ymin+0.02, ymax=1, color='black', lw=1, ls=':') # draw vertical line
        ax.scatter(v, ymin, color=colors[1], marker='o', s=30)
        text = str(round(ymin, 2))
        if 0 < ymin < 0.01:
            ax.text(x=v-(65*5), y=1.05, s=f"{ymin:.0e}", fontdict={'fontsize':11})
        else:
            if len(re.search('(?<=\.).+', str(text)).group(0)) == 1:
                ax.text(x=v-(56*5), y=1.05, s=text+'0', fontdict={'fontsize':11})
            else:
                ax.text(x=v-(56*5), y=1.05, s=text, fontdict={'fontsize':11})
        for n in range(2):
            row = test_df.iloc[v+n]
            ind = row_counts.index.to_list().index(row.data) # get index of dataset w/in value counts
            floor = sum([row_counts.iloc[i] for i in range(ind)]) # get the number of preceding images in prob_df 
            img_loc = row.name - floor # gets index of original test image to transform
            img = datasets[row.data][img_loc][0] # pulls the transformed image
            y0 = 1.1 + (n * .145)
            new_ax = ax.inset_axes([x0, y0, 666, .15], transform=ax.transData) # create new inset axes in data coordinates
            new_ax.imshow(img.reshape(28,28), cmap='gray') # plot image
            new_ax.set_xticks([])
            new_ax.set_yticks([])
    
    ax.set_xlabel('Image Quantile', fontsize=13, labelpad=7.5)
    ax.set_ylim([0, 1])
    #ax.set_xlim([-0.05, 1.05])
    ax.set_ylabel('SAGE Score', fontsize=13, labelpad=7.5)
    ax.tick_params(axis='both', which='major', labelsize=11)
    if save == True:
        joined_path = os.path.join(save_path + f'test_prob_quantiles.png')
        plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def test_measure_quantiles(probs_df, datasets, save=False, save_path=None):
    # set up image variables
    test_df = probs_df.loc[probs_df.data == 'test']
    meas_dict = {
        'dist_prob':'Distance Probability',
        'rloss_prob':'Reconstruction Probability',
        'task_prob':'Classifier Probability'
    }
    colors = sns.color_palette("colorblind").as_hex()
    for meas in ['dist_prob', 'rloss_prob', 'task_prob']:
        meas_samp = test_df[['data', meas]].sort_values(meas)
        counts = list(range(1, len(meas_samp)+1))
        row_counts = probs_df.data.value_counts()
        target_values = np.append(np.linspace(0, 7999, 11, dtype=int)[:-1], 7998)
        width, length = 666, 666
        
        fig, ax = plt.subplots(figsize=(7, 4))
        # plot non-exceedance curves
        #cumulative_probs = [1 - (j+1)/len(train_df) for j, _ in enumerate(np.sort(train_df.gmean_metric.to_numpy()))]
        #sorted_probs = np.sort(train_df.gmean_metric.to_numpy())
        scores = meas_samp[meas].to_numpy() # scores are pre-sorted
        ax.plot(counts, scores, color=colors[1], lw=3)
        #ax.fill_between(sorted_probs, cumulative_probs, color='dimgrey', alpha=0.5)
        
        # plot example images above the hist
        for j, v in enumerate(target_values):
            x0 = v - 333
            ymin = scores[v]
            ax.axvline(x=v, ymin=ymin+0.02, ymax=1, color='black', lw=1, ls=':') # draw vertical line
            ax.scatter(v, ymin, color=colors[1], marker='o', s=30)
            text = str(round(ymin, 2))
            if 0 < ymin < 0.01:
                ax.text(x=v-(65*5), y=1.05, s=f"{ymin:.0e}", fontdict={'fontsize':11})
            else:
                if len(re.search('(?<=\.).+', str(text)).group(0)) == 1:
                    ax.text(x=v-(56*5), y=1.05, s=text+'0', fontdict={'fontsize':11})
                else:
                    ax.text(x=v-(56*5), y=1.05, s=text, fontdict={'fontsize':11})
            for n in range(2):
                row = meas_samp.iloc[v+n]
                ind = row_counts.index.to_list().index(row.data) # get index of dataset w/in value counts
                floor = sum([row_counts.iloc[i] for i in range(ind)]) # get the number of preceding images in prob_df 
                img_loc = row.name - floor # gets index of original test image to transform
                img = datasets[row.data][img_loc][0] # pulls the transformed image
                y0 = 1.1 + (n * .145)
                new_ax = ax.inset_axes([x0, y0, 666, .15], transform=ax.transData) # create new inset axes in data coordinates
                new_ax.imshow(img.reshape(28,28), cmap='gray') # plot image
                new_ax.set_xticks([])
                new_ax.set_yticks([])
        
        ax.set_xlabel('Image Quantiles', fontsize=13, labelpad=7.5)
        #ax.set_ylim([-0.05, 1.05])
        ax.set_ylim([0, 1])
        ax.set_ylabel(meas_dict[meas], fontsize=13, labelpad=7.5)
        ax.tick_params(axis='both', which='major', labelsize=11)
        if save == True:
            joined_path = os.path.join(save_path + f'test_{meas}_quantiles.png')
            plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()

def accuracy_vs_proportion(probs_df, data_type=None, save=False, save_path=None):
    assert len(probs_df.dim.unique()) == 1
    dim = probs_df.dim.unique()[0]
    thresh = np.linspace(0, 0.8, 50)
    for name in ['train', 'test', 'transform']:
        # get accuracy change over threshold filter
        acc_arr = []
        # get remaining n samples over threshold filter
        examp_arr = []
        if name == 'transform':
            sub_df = probs_df.loc[(probs_df.data != 'train') & (probs_df.data != 'test')]
            if data_type == 'MNIST':
                # remove horizontal and vertical flips for 2 and 5 images
                sub_df = sub_df.loc[~((sub_df['labels'].isin([2, 5])) & (sub_df['data'].isin(['horizontal', 'vertical'])))]
        else:
            sub_df = probs_df.loc[probs_df['data'] == name]
        for t in thresh:
            thresh_df = sub_df.loc[sub_df.gmean_metric >= t]
            acc = thresh_df.resnet_correct.sum()/thresh_df.shape[0]
            acc_arr.append(acc)
            examp = thresh_df.shape[0] / sub_df.shape[0]
            examp_arr.append(examp)
        fig, ax = plt.subplots()
        # make elbow plot for accuracy
        ax.plot(thresh, acc_arr, marker="o", markersize=3, color='navy', lw=2)
        ax.set_xlabel('SAGE Threshold', labelpad=10, fontsize=11)
        ax.set_ylabel('ResNet Accuracy', color='navy', labelpad=10, fontsize=11)
        ax.set_title(f'{data_type} {name.capitalize()}', fontsize=12)
        # make elbow plot for sample filter
        ax2 = ax.twinx()
        ax2.plot(thresh, examp_arr, marker="o", markersize=3, color='firebrick', lw=2)
        ax2.set_ylabel('Proportion Data Remaining', color='firebrick', labelpad=10, fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax2.tick_params(axis='both', which='major', labelsize=11)
        if save == True:
            joined_path = os.path.join(save_path + f'{name}_accvsprop.png')
            plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
        plt.close()

def scorefilter_curves(probs_df, name_dict, data_type=None, save=False, save_path=None):
    colors = sns.color_palette("colorblind").as_hex()
    marks = ['o', 'X', 'v', '^', '*', 'D', 'h', 'H', 'P', 's', 'd']
    thresh = np.linspace(0, 0.2, 5)
    thresh = np.insert(thresh, 1, 0.01)
    for i, name in enumerate(['train', 'test', 'transform']):
        fig, ax = plt.subplots()
        if name == 'transform':
            sub_df = probs_df.loc[(probs_df.data != 'train') & (probs_df.data != 'test')]
            if data_type == 'MNIST':
                # remove horizontal and vertical flips for 2 and 5 images
                sub_df = sub_df.loc[~((sub_df['labels'].isin([2, 5])) & (sub_df['data'].isin(['horizontal', 'vertical'])))]
            # get remaining n samples over threshold filter
            for j, label in enumerate(sub_df['data'].unique()):
                remain_arr = []
                name_df = sub_df.loc[sub_df['data'] == label]
                for t in thresh:
                    left = name_df.loc[name_df.gmean_metric >= t]
                    n_left = left.shape[0]
                    prop_left = n_left / name_df.shape[0]
                    remain_arr.append(prop_left)
                ax.plot(
                    thresh,
                    remain_arr, 
                    marker=marks[j], 
                    markersize=9.5, 
                    markerfacecolor=colors[i],
                    markeredgecolor='black',
                    color=colors[2],
                    lw=2.5, 
                    label=name_dict[label], 
                    alpha=0.8
                )
            # plot line tracing overall proportion
            overall_remain = []
            for t in thresh:
                overall_left = sub_df.loc[sub_df.gmean_metric >= t]
                n_left = overall_left.shape[0]
                prop_left = n_left / sub_df.shape[0]
                overall_remain.append(prop_left)
            ax.plot(thresh, overall_remain, color='black', linestyle='--', lw=2.5, label='Total', alpha=0.75)
            ax.set_xlabel('SAGE Threshold', fontsize=13)
            ax.set_ylabel('Proportion Remaining', fontsize=13)
            ax.set_ylim([-0.05, 1.05])
            ax.tick_params(axis='both', which='major', labelsize=11)
            ax.legend(bbox_to_anchor=(1,1), title='Transform Applied', fontsize=11, title_fontsize=11)
            plt.grid(True)
            if save == True:
                joined_path = os.path.join(save_path + f'{name}_scorefilter_curve.png')
                plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()
        else:
            sub_df = probs_df.loc[probs_df['data'] == name]
            # get remaining n samples over threshold filter
            remain_arr = []
            for t in thresh:
                left = sub_df.loc[sub_df.gmean_metric >= t]
                n_left = left.shape[0]
                prop_left = n_left / sub_df.shape[0]
                remain_arr.append(prop_left)
            ax.plot(
                thresh, 
                remain_arr, 
                marker='o',
                markersize=7, 
                color=colors[i],
                lw=3, 
                label=f'{name.capitalize()}', 
                alpha=0.9
            )
            ax.set_xlabel('SAGE Threshold', fontsize=13)
            ax.set_ylabel('Proportion Remaining', fontsize=13)
            ax.set_ylim([-0.05, 1.05])
            ax.tick_params(axis='both', which='major', labelsize=11)
            plt.grid(True)
            if save == True:
                joined_path = os.path.join(save_path + f'{name}_scorefilter_curve.png')
                plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()
    
def resnet_prcurves(resnet_df, probs_df, data_type=None, save=False, save_path=None):
    thresh = np.linspace(0, 0.2, 5)
    thresh = np.insert(thresh, 1, 0.01)
    train_labels = resnet_df.loc[resnet_df.data == 'train'].labels.to_numpy()
    binarizer = LabelBinarizer().fit(train_labels)
    conf_cols = [f'conf{i}' for i in range(10)]
    for i, name in enumerate(['train', 'test', 'transform']):
        if name == 'train':
            colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.8, len(thresh)))
            sub_df = probs_df.loc[probs_df['data'] == 'train']
            fig, ax = plt.subplots()
            for j, t in enumerate(thresh):
                t = round(t, 3)
                cutoff_df = sub_df.loc[sub_df.gmean_metric >= t]
                cut_inds = cutoff_df.index.to_numpy()
                cutoff_labels = resnet_df.iloc[cut_inds].labels.to_numpy()
                cutoff_labels_onehot = binarizer.transform(cutoff_labels)
                cutoff_probs = resnet_df.iloc[cut_inds][conf_cols].to_numpy()
                display = PrecisionRecallDisplay.from_predictions(
                    cutoff_labels_onehot.ravel(),
                    cutoff_probs.ravel(),
                    name=t,
                    color=colors[j],
                    plot_chance_level=False,
                    ax=ax
                )
            ax.set_xlabel("Recall", fontsize=13)
            ax.set_ylabel("Precision", fontsize=13)
            ax.legend(title='Threshold Level', loc='lower left', fontsize=11, title_fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=11)
            if save == True:
                joined_path = os.path.join(save_path + f'{name}_prcurves.png')
                plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()

        elif name == 'test':
            colors = plt.get_cmap('Oranges')(np.linspace(0.2, 0.8, len(thresh)))
            sub_df = probs_df.loc[probs_df['data'] == 'test']
            fig, ax = plt.subplots()
            for j, t in enumerate(thresh):
                t = round(t, 3)
                cutoff_df = sub_df.loc[sub_df.gmean_metric >= t]
                cut_inds = cutoff_df.index.to_numpy()
                cutoff_labels = resnet_df.iloc[cut_inds].labels.to_numpy()
                cutoff_labels_onehot = binarizer.transform(cutoff_labels)
                cutoff_probs = resnet_df.iloc[cut_inds][conf_cols].to_numpy()
                display = PrecisionRecallDisplay.from_predictions(
                    cutoff_labels_onehot.ravel(),
                    cutoff_probs.ravel(),
                    name=t,
                    color=colors[j],
                    plot_chance_level=False,
                    ax=ax
                )
            ax.set_xlabel("Recall", fontsize=13)
            ax.set_ylabel("Precision", fontsize=13)
            ax.legend(title='Threshold Level', loc='lower left', fontsize=11, title_fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=11)
            if save == True:
                joined_path = os.path.join(save_path + f'{name}_prcurves.png')
                plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
        elif name == 'transform':
            colors = plt.get_cmap('Greens')(np.linspace(0.2, 0.8, len(thresh)))
            sub_df = probs_df.loc[(probs_df.data != 'train') & (probs_df.data != 'test')]
            if data_type == 'MNIST':
                # remove horizontal and vertical flips for 2 and 5 images
                sub_df = sub_df.loc[~((sub_df['labels'].isin([2, 5])) & (sub_df['data'].isin(['horizontal', 'vertical'])))]
            fig, ax = plt.subplots()
            for j, t in enumerate(thresh):
                t = round(t, 3)
                cutoff_df = sub_df.loc[sub_df.gmean_metric >= t]
                cut_inds = cutoff_df.index.to_numpy()
                cutoff_labels = resnet_df.iloc[cut_inds].labels.to_numpy()
                cutoff_labels_onehot = binarizer.transform(cutoff_labels)
                cutoff_probs = resnet_df.iloc[cut_inds][conf_cols].to_numpy()
                display = PrecisionRecallDisplay.from_predictions(
                    cutoff_labels_onehot.ravel(),
                    cutoff_probs.ravel(),
                    name=t,
                    color=colors[j],
                    plot_chance_level=False,
                    ax=ax
                )
            ax.set_xlabel("Recall", fontsize=13)
            ax.set_ylabel("Precision", fontsize=13)
            ax.legend(title='Threshold Level', loc='lower left', fontsize=11, title_fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=11)
            if save == True:
                joined_path = os.path.join(save_path + f'{name}_prcurves.png')
                plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()

def resnet_altprobs_prcurves(resnet_df, probs_df, data_type=None, save=False, save_path=None):
    p_dict = {
        'dist_prob':'kNN Distance Probability', 
        'rloss_prob':'Reconstruction Error Probability', 
        'task_prob': 'Classifier Confidence Probability'
    }
    thresh = np.linspace(0, 0.2, 5)
    thresh = np.insert(thresh, 1, 0.01)
    train_labels = resnet_df.loc[resnet_df.data == 'train'].labels.to_numpy()
    binarizer = LabelBinarizer().fit(train_labels)
    conf_cols = [f'conf{i}' for i in range(10)]
    colors = plt.get_cmap('Greens')(np.linspace(0.2, 0.8, len(thresh)))
    sub_df = probs_df.loc[~probs_df.data.isin(['train', 'test'])]
    if data_type == 'MNIST':
        # remove horizontal and vertical flips for 2 and 5 images
        sub_df = sub_df.loc[~((sub_df['labels'].isin([2, 5])) & (sub_df['data'].isin(['horizontal', 'vertical'])))]
    for p in ['dist_prob', 'rloss_prob', 'task_prob']:
        fig, ax = plt.subplots()
        for j, t in enumerate(thresh):
            t = round(t, 3)
            cutoff_df = sub_df.loc[sub_df[p] >= t]
            cut_inds = cutoff_df.index.to_numpy()
            cutoff_labels = resnet_df.iloc[cut_inds].labels.to_numpy()
            cutoff_labels_onehot = binarizer.transform(cutoff_labels)
            cutoff_probs = resnet_df.iloc[cut_inds][conf_cols].to_numpy()
            display = PrecisionRecallDisplay.from_predictions(
                cutoff_labels_onehot.ravel(),
                cutoff_probs.ravel(),
                name=t,
                color=colors[j],
                plot_chance_level=False,
                ax=ax
            )
        ax.set_xlabel("Recall", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        ax.legend(title='Threshold Level', loc='lower left', fontsize=11, title_fontsize=11)
        ax.set_title(p_dict[p], fontsize=13)
        ax.tick_params(axis='both', which='major', labelsize=11)
        if save == True:
            joined_path = os.path.join(save_path + f'trans_{p}_prcurves.png')
            plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
        plt.close()

def resnet_roccurves(resnet_df, probs_df, data_type=None, save=False, save_path=None):
    thresh = np.linspace(0, 0.2, 5)
    thresh = np.insert(thresh, 1, 0.01)
    train_labels = resnet_df.loc[resnet_df.data == 'train'].labels.to_numpy()
    binarizer = LabelBinarizer().fit(train_labels)
    conf_cols = [f'conf{i}' for i in range(10)]
    for i, name in enumerate(['train', 'test', 'transform']):
        if name == 'train':
            colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.8, len(thresh)))
            sub_df = probs_df.loc[probs_df['data'] == 'train']
            fig, ax = plt.subplots()
            for j, t in enumerate(thresh):
                t = round(t, 3)
                cutoff_df = sub_df.loc[sub_df.gmean_metric >= t]
                cut_inds = cutoff_df.index.to_numpy()
                cutoff_labels = resnet_df.iloc[cut_inds].labels.to_numpy()
                cutoff_labels_onehot = binarizer.transform(cutoff_labels)
                cutoff_probs = resnet_df.iloc[cut_inds][conf_cols].to_numpy()
                display = RocCurveDisplay.from_predictions(
                    cutoff_labels_onehot.ravel(),
                    cutoff_probs.ravel(),
                    name=t,
                    color=colors[j],
                    plot_chance_level=False,
                    ax=ax
                )
            ax.set_xlabel("False Positive Rate", fontsize=13)
            ax.set_ylabel("True Positive Rate", fontsize=13)
            ax.legend(title='Threshold Level', loc='lower left', fontsize=11, title_fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=11)
            if save == True:
                joined_path = os.path.join(save_path + f'{name}_roccurves.png')
                plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()

        elif name == 'test':
            colors = plt.get_cmap('Oranges')(np.linspace(0.2, 0.8, len(thresh)))
            sub_df = probs_df.loc[probs_df['data'] == 'test']
            fig, ax = plt.subplots()
            for j, t in enumerate(thresh):
                t = round(t, 3)
                cutoff_df = sub_df.loc[sub_df.gmean_metric >= t]
                cut_inds = cutoff_df.index.to_numpy()
                cutoff_labels = resnet_df.iloc[cut_inds].labels.to_numpy()
                cutoff_labels_onehot = binarizer.transform(cutoff_labels)
                cutoff_probs = resnet_df.iloc[cut_inds][conf_cols].to_numpy()
                display = RocCurveDisplay.from_predictions(
                    cutoff_labels_onehot.ravel(),
                    cutoff_probs.ravel(),
                    name=t,
                    color=colors[j],
                    plot_chance_level=False,
                    ax=ax
                )
            ax.set_xlabel("False Positive Rate", fontsize=13)
            ax.set_ylabel("True Positive Rate", fontsize=13)
            ax.legend(title='Threshold Level', loc='lower left', fontsize=11, title_fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=11)
            if save == True:
                joined_path = os.path.join(save_path + f'{name}_roccurves.png')
                plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
        elif name == 'transform':
            colors = plt.get_cmap('Greens')(np.linspace(0.2, 0.8, len(thresh)))
            sub_df = probs_df.loc[(probs_df.data != 'train') & (probs_df.data != 'test')]
            if data_type == 'MNIST':
                # remove horizontal and vertical flips for 2 and 5 images
                sub_df = sub_df.loc[~((sub_df['labels'].isin([2, 5])) & (sub_df['data'].isin(['horizontal', 'vertical'])))]
            fig, ax = plt.subplots()
            for j, t in enumerate(thresh):
                t = round(t, 3)
                cutoff_df = sub_df.loc[sub_df.gmean_metric >= t]
                cut_inds = cutoff_df.index.to_numpy()
                cutoff_labels = resnet_df.iloc[cut_inds].labels.to_numpy()
                cutoff_labels_onehot = binarizer.transform(cutoff_labels)
                cutoff_probs = resnet_df.iloc[cut_inds][conf_cols].to_numpy()
                display = RocCurveDisplay.from_predictions(
                    cutoff_labels_onehot.ravel(),
                    cutoff_probs.ravel(),
                    name=t,
                    color=colors[j],
                    plot_chance_level=False,
                    ax=ax
                )
            ax.set_xlabel("False Positive Rate", fontsize=13)
            ax.set_ylabel("True Positive Rate", fontsize=13)
            ax.legend(title='Threshold Level', loc='lower left', fontsize=11, title_fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=11)
            if save == True:
                joined_path = os.path.join(save_path + f'{name}_roccurves.png')
                plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()

def mnist_imposter_scorefilter(probs_df, save=False, save_path=None):
    colors = sns.color_palette("colorblind").as_hex()
    marks = ['o', 'X', 'v', '^', '*', 'D', 'h', 'H', 'P', 's', 'd']
    thresh = np.linspace(0, 0.2, 5)
    thresh = np.insert(thresh, 1, 0.01)
    fig, ax = plt.subplots()
    sub_df = probs_df.loc[(probs_df.data != 'train') & (probs_df.data != 'test')] # exclude train and test
    # isolate horizontal and vertical flips for 2 and 5 images
    imposter_df = sub_df.loc[((sub_df['labels'].isin([2, 5])) & (sub_df['data'].isin(['horizontal', 'vertical'])))]
    # remove horizontal and vertical flips for 2 and 5 images
    other_df = sub_df.loc[~((sub_df['labels'].isin([2, 5])) & (sub_df['data'].isin(['horizontal', 'vertical'])))]
    # isolate classes with same appearance after flips
    syn_df = other_df.loc[other_df.labels.isin([0, 1, 3, 8])]
    syn_df = syn_df.loc[~((syn_df['labels'].isin([3])) & (syn_df['data'].isin(['horizontal'])))]
    # isolate classes with different appearance after flips
    nonsyn_df = other_df.loc[other_df.labels.isin([3, 4, 6, 7, 9])]
    nonsyn_df = nonsyn_df.loc[~((nonsyn_df['labels'] == 3) & (nonsyn_df['data'] == 'vertical'))]
    for i, data in enumerate(['imposter', 'synonymous', 'nonsynonymous']):
        if data == 'imposter':
            df = imposter_df
        elif data == 'synonymous':
            df = syn_df
        else:
            df = nonsyn_df
        for j, flip in enumerate(['horizontal', 'vertical']):
            remain_arr = []
            flip_df = df.loc[df['data'] == flip]
            for t in thresh:
                left = flip_df.loc[flip_df.gmean_metric >= t]
                n_left = left.shape[0]
                prop_left = n_left / flip_df.shape[0]
                remain_arr.append(prop_left)
            ax.plot(
                thresh,
                remain_arr, 
                marker=marks[j], 
                markersize=9.5, 
                markerfacecolor=colors[i + 6],
                markeredgecolor='black',
                color=colors[i + 6],
                lw=2.5, 
                label=f'{flip.capitalize()} – {data.capitalize()}', 
                alpha=0.8
            )
    ax.set_xlabel('SAGE Threshold', fontsize=13)
    ax.set_ylabel('Proportion Remaining', fontsize=13)
    ax.set_ylim([-0.05, 1.05])
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(bbox_to_anchor=(1,1), title='Transform', fontsize=11, title_fontsize=11)
    plt.grid(True)
    if save == True:
        joined_path = os.path.join(save_path + f'imposter_scorefilter_curve.png')
        plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
    plt.close()

def mnist_imposter_prcurves(probs_df, resnet_df, save=False, save_path=None):
    thresh = np.linspace(0, 0.2, 5)
    thresh = np.insert(thresh, 1, 0.01)
    train_labels = resnet_df.loc[resnet_df.data == 'train'].labels.to_numpy()
    binarizer = LabelBinarizer().fit(train_labels)
    conf_cols = [f'conf{i}' for i in range(10)]
    sub_df = probs_df.loc[(probs_df.data != 'train') & (probs_df.data != 'test')] # exclude train and test
    # isolate horizontal and vertical flips for 2 and 5 images
    imposter_df = sub_df.loc[((sub_df['labels'].isin([2, 5])) & (sub_df['data'].isin(['horizontal', 'vertical'])))]
    # remove horizontal and vertical flips for 2 and 5 images
    other_df = sub_df.loc[~((sub_df['labels'].isin([2, 5])) & (sub_df['data'].isin(['horizontal', 'vertical'])))]
    # isolate classes with same appearance after flips
    syn_df = other_df.loc[other_df.labels.isin([0, 1, 3, 8])]
    syn_df = syn_df.loc[~((syn_df['labels'].isin([3])) & (syn_df['data'].isin(['horizontal'])))]
    # isolate classes with different appearance after flips
    nonsyn_df = other_df.loc[other_df.labels.isin([3, 4, 6, 7, 9])]
    nonsyn_df = nonsyn_df.loc[~((nonsyn_df['labels'] == 3) & (nonsyn_df['data'] == 'vertical'))]
    for i, data in enumerate(['imposter', 'synonymous', 'nonsynonymous']):
        if data == 'imposter':
            df = imposter_df
            colors = plt.get_cmap('RdPu')(np.linspace(0.1, 0.5, len(thresh)))
            fig, ax = plt.subplots()
            for j, t in enumerate(thresh):
                t = round(t, 3)
                cutoff_df = df.loc[df.gmean_metric >= t]
                cut_inds = cutoff_df.index.to_numpy()
                cutoff_labels = resnet_df.iloc[cut_inds].labels.to_numpy()
                cutoff_labels_onehot = binarizer.transform(cutoff_labels)
                cutoff_probs = resnet_df.iloc[cut_inds][conf_cols].to_numpy()
                display = PrecisionRecallDisplay.from_predictions(
                    cutoff_labels_onehot.ravel(),
                    cutoff_probs.ravel(),
                    name=t,
                    color=colors[j],
                    plot_chance_level=False,
                    ax=ax
                )
            ax.set_xlabel("Recall", fontsize=13)
            ax.set_ylabel("Precision", fontsize=13)
            ax.legend(title='Threshold Level', loc='lower left', fontsize=11, title_fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=11)
            if save == True:
                joined_path = os.path.join(save_path + f'{data}_prcurves.png')
                plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()
        elif data == 'synonymous':
            df = syn_df
            colors = plt.get_cmap('Greys')(np.linspace(0.1, 0.5, len(thresh)))
            fig, ax = plt.subplots()
            for j, t in enumerate(thresh):
                t = round(t, 3)
                cutoff_df = df.loc[df.gmean_metric >= t]
                cut_inds = cutoff_df.index.to_numpy()
                cutoff_labels = resnet_df.iloc[cut_inds].labels.to_numpy()
                cutoff_labels_onehot = binarizer.transform(cutoff_labels)
                cutoff_probs = resnet_df.iloc[cut_inds][conf_cols].to_numpy()
                display = PrecisionRecallDisplay.from_predictions(
                    cutoff_labels_onehot.ravel(),
                    cutoff_probs.ravel(),
                    name=t,
                    color=colors[j],
                    plot_chance_level=False,
                    ax=ax
                )
            ax.set_xlabel("Recall", fontsize=13)
            ax.set_ylabel("Precision", fontsize=13)
            ax.legend(title='Threshold Level', loc='lower left', fontsize=11, title_fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=11)
            if save == True:
                joined_path = os.path.join(save_path + f'{data}_prcurves.png')
                plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()
        else:
            df = nonsyn_df
            colors = ['lightyellow', 'gold', 'goldenrod']
            cmap_yellow = mpl.colors.LinearSegmentedColormap.from_list('custom_yellow', colors)
            colors = cmap_yellow(np.linspace(0.2, 0.8, len(thresh)))
            fig, ax = plt.subplots()
            for j, t in enumerate(thresh):
                t = round(t, 3)
                cutoff_df = df.loc[df.gmean_metric >= t]
                cut_inds = cutoff_df.index.to_numpy()
                cutoff_labels = resnet_df.iloc[cut_inds].labels.to_numpy()
                cutoff_labels_onehot = binarizer.transform(cutoff_labels)
                cutoff_probs = resnet_df.iloc[cut_inds][conf_cols].to_numpy()
                display = PrecisionRecallDisplay.from_predictions(
                    cutoff_labels_onehot.ravel(),
                    cutoff_probs.ravel(),
                    name=t,
                    color=colors[j],
                    plot_chance_level=False,
                    ax=ax
                )
            ax.set_xlabel("Recall", fontsize=13)
            ax.set_ylabel("Precision", fontsize=13)
            ax.legend(title='Threshold Level', loc='lower left', fontsize=11, title_fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=11)
            if save == True:
                joined_path = os.path.join(save_path + f'{data}_prcurves.png')
                plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()

def abalone_latent1d_stripplot(latent_df, name_dict, save=False, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.stripplot(
        data=latent_df, 
        x="latent1", 
        y="data", 
        hue="labels",
        palette="inferno_r",
        jitter=0.15,
        dodge=False, 
        alpha=0.75, 
        legend=False, 
        edgecolor='black',
        ax=ax
    )
    # Make a colorbar
    cmap = plt.get_cmap("inferno")
    norm = plt.Normalize(latent_df['labels'].min(), latent_df['labels'].max())
    sm =  plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('# Rings', rotation=90, fontsize=13, labelpad=10)
    cbar.ax.tick_params(labelsize=11)
    # Change label details
    ax.set_xlabel('Latent Dimension 1', fontsize=13, labelpad=10)
    ax.set_ylabel('')
    ax.set_yticks(list(range(len(latent_df['data'].unique()))))
    ax.set_yticklabels([name_dict[d] for d in latent_df['data'].unique()])
    ax.tick_params(axis='x', which='major', labelsize=11)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.set_xlim([-7, 27])
    if save == True:
        joined_path = os.path.join(save_path, 'latent1d_stripplot.png')
        plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def abalone_output_boxplots(latent_df, name_dict, save=False, save_path=None):
    meas_dict = {
        'dist':'kNN Distance to Train',
        'rloss':'Reconstruction Error',
        'task':'Regression Error',
    }
    assert len(latent_df.dim.unique()) == 1
    dim = latent_df.dim.unique()[0]
    # make boxplots
    plot_df = latent_df.loc[latent_df['data'].isin(name_dict.keys())]
    plot_df['data'] = plot_df['data'].replace(name_dict)
    for meas in ['dist', 'rloss', 'task']:
        fig, ax = plt.subplots()
        sns.boxplot(
            data=plot_df, 
            x='data', 
            y=meas, 
            color='lightsteelblue', 
            showfliers=False, 
            ax=ax, 
            boxprops={'edgecolor': 'black'},
            whiskerprops={'color':'black'}, 
            capprops={'color':'black'}, 
            medianprops={'color':'black'}
        )
        ax.tick_params(axis='x', labelsize=9)
        ax.set_ylabel(meas_dict[meas], labelpad=10, fontsize=11)
        ax.set_xlabel('')
        plt.yticks(fontsize=11)
        if save == True:
            joined_path = os.path.join(save_path, f'abalone_regressorsae{dim}D_{meas}_boxplots.png')
            plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
        plt.close()
    
def abalone_score_boxplots(probs_df, name_dict, save=False, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette("colorblind").as_hex()
    ordered_colors = [colors[0], colors[1]] + [colors[2]]*len([n for n in probs_df['data'].unique() if n not in ['train', 'test']])
    latent = []
    for name in probs_df['data'].unique():
        latent.append(probs_df.loc[probs_df.data == name].gmean_metric.to_numpy())
    bplot = ax.boxplot(
        latent, 
        tick_labels=[name_dict[i] for i in probs_df['data'].unique()],
        widths=0.65,
        vert=False,
        patch_artist=True,
        showfliers=False
    )
    for p, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(ordered_colors[p])
        patch.set_alpha(0.75)
    for median in bplot['medians']:
        median.set_color('black')
    #for i, c in enumerate(ordered_colors):
        #ax.scatter([i+1]*len(latent[i]), latent[i], color=c, s=30, alpha=0.3)
    ax.set_xlabel('SAGE Score', fontsize=13, labelpad=10)
    ax.tick_params(axis='x', which='major', labelsize=11)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.invert_yaxis()
    if save == True:
        joined_path = os.path.join(save_path, 'abalone_score_boxplots.png')
        plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def abalone_score_violinplots(probs_df, name_dict, save=False, save_path=None):
    fig, ax = plt.subplots(figsize=(7,5))
    colors = sns.color_palette("colorblind").as_hex()
    ordered_colors = [colors[0], colors[1]] + [colors[2]]*len([n for n in probs_df['data'].unique() if n not in ['train', 'test']])
    scores = []
    for name in probs_df['data'].unique():
        scores.append(probs_df.loc[probs_df.data == name].gmean_metric.to_numpy())
    bplot = ax.boxplot(
        scores, 
        tick_labels=[name_dict[i] for i in probs_df['data'].unique()],
        widths=0.35,
        vert=False,
        patch_artist=True,
        showfliers=False
    )
    for patch in bplot['boxes']:
        patch.set_facecolor('white')
        patch.set_alpha(0.75)
    for median in bplot['medians']:
        median.set_color('black')
    parts = ax.violinplot(scores, vert=False, showmeans=False, showmedians=False, showextrema=False)
    for pc, color in zip(parts['bodies'], ordered_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.75)
    ax.set_xlabel('SAGE Score', fontsize=13, labelpad=10)
    ax.invert_yaxis()
    ax.tick_params(axis='x', which='major', labelsize=11)
    ax.tick_params(axis='y', which='major', labelsize=11)
    if save == True:
        joined_path = os.path.join(save_path, 'abalone_score_violinplots.png')
        plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def abalone_feature_importance(regressor, data_dict, original_columns, save=False, save_path=None):
    train_X = data_dict['train'].data.numpy()
    train_y = data_dict['train'].targets.numpy().ravel()
    
    mdi_importances = pd.Series(regressor.best_estimator_.feature_importances_, index=original_columns)
    tree_importance_sorted_idx = np.argsort(regressor.best_estimator_.feature_importances_)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    mdi_importances.sort_values().plot.barh(ax=ax1)
    ax1.set_xlabel("Gini importance")
    
    result = permutation_importance(regressor, train_X, train_y, n_repeats=20, scoring='r2', random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()
    tick_labels_dict = {'tick_labels': np.array(original_columns)[perm_sorted_idx]}
    ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False, **tick_labels_dict)
    ax2.axvline(x=0, color="k", linestyle="--")
    ax2.set_xlabel("Decrease in r2 score")
    fig.suptitle(
        "Impurity-based vs. permutation importances on Abalone Train"
    )
    _ = fig.tight_layout()
    if save == True:
        joined_path = os.path.join(save_path, 'abalone_feature_importances.png')
        plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def abalone_accuracy_vs_proportion(preds_df, dim, save=False, save_path=None):
    thresh = np.linspace(0, 0.7, 15)
    for name in ['train', 'test', 'transform']:
        errors, proportions = [], []
        if name == 'transform':
            sub_df = preds_df.loc[(preds_df.data != 'train') & (preds_df.data != 'test')]
        else:
            sub_df = preds_df.loc[preds_df.data == name]
        for t in thresh:
            cutoff = sub_df.loc[sub_df.gmean_metric >= t]
            # get prediction errors
            rmse = root_mean_squared_error(cutoff.Rings, cutoff.preds)
            errors.append(rmse)
            # get proportion of data remaining
            prop = len(cutoff) / len(sub_df)
            proportions.append(prop)
        
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        # make elbow plot for accuracy
        ax.plot(thresh, errors, marker="o", markersize=5, color='navy', lw=2, alpha=0.75)
        # make elbow plot for proportion remaining
        ax2.plot(thresh, proportions, marker="o", markersize=5, color='firebrick', lw=2, alpha=0.75)
        ax.set_ylabel('Root Mean Squared Error (RMSE)', fontsize=13, color='navy', rotation='vertical', verticalalignment='center', labelpad=20)
        ax2.set_ylabel('Proportion Remaining', fontsize=13, color='firebrick', rotation='vertical', verticalalignment='center', labelpad=20)
        ax.set_xlabel('SAGE Threshold', fontsize=13, labelpad=10)
        ax.tick_params(axis='y', which='major', labelsize=11, colors='navy')
        ax.tick_params(axis='x', which='major', labelsize=11)
        ax2.tick_params(axis='y', which='major', labelsize=11, colors='firebrick')
        plt.title(f'Abalone {name.capitalize()}', fontsize=13)
        if save == True:
            joined_path = os.path.join(save_path + f'{name}_accvsprop.png')
            plt.savefig(joined_path, format='png', dpi=150, bbox_inches='tight')
        plt.close()

def abalone_actual_vs_pred(preds_df, save=False, save_path=None):
    thresh = np.linspace(0, 0.2, 5)
    thresh = np.insert(thresh, 1, 0.01)
    color_list = ['Blues', 'Oranges', 'Greens']
    for i, name in enumerate(['train', 'test', 'transform']):
        fig, ax = plt.subplots()
        colors = plt.get_cmap(color_list[i])(np.linspace(0.1, 0.8, len(thresh)))
        if name == 'transform':
            sub_df = preds_df.loc[(preds_df['data'] != 'train') & (preds_df['data'] != 'test')]
            for j, t in enumerate(thresh):
                thresh_df = sub_df.loc[sub_df.gmean_metric >= t]
                rmse = root_mean_squared_error(thresh_df.Rings.to_numpy(), thresh_df.preds.to_numpy())
                ax.scatter(thresh_df.preds, thresh_df.Rings, s=40, label=f'{round(t, 2)} (RMSE = {rmse:.2f})', color=colors[j], alpha=0.3)
            ax.set_xlabel('Predicted Rings', fontsize=13)
            ax.set_ylabel('Actual Rings', fontsize=13)
            ax.legend(title='Threshold Level', fontsize=11, title_fontsize=11, loc='lower right')
            ax.tick_params(axis='both', which='major', labelsize=11)
            ax.plot([41,0], [41,0], color='black', ls=':', alpha=0.75)
            ax.set_xlim([-1, 41])
            ax.set_ylim([-1, 41])
            if save == True:
                joined_path = os.path.join(save_path, 'abalone_trans_preds_vs_actual.png')
                plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
            plt.close()
        elif name == 'train':
            sub_df = preds_df.loc[preds_df['data'] == 'train']
            for j, t in enumerate(thresh):
                thresh_df = sub_df.loc[sub_df.gmean_metric >= t]
                rmse = root_mean_squared_error(thresh_df.Rings, thresh_df.preds)
                ax.scatter(thresh_df.preds, thresh_df.Rings, s=40, label=f'{round(t, 2)} (RMSE = {rmse:.2f})', color=colors[j], alpha=0.3)
            ax.set_xlabel('Predicted Rings', fontsize=13)
            ax.set_ylabel('Actual Rings', fontsize=13)
            ax.legend(title='Threshold Level', fontsize=11, title_fontsize=11, loc='lower right')
            ax.tick_params(axis='both', which='major', labelsize=11)
            ax.plot([41,0], [41,0], color='black', ls=':', alpha=0.75)
            ax.set_xlim([-1, 41])
            ax.set_ylim([-1, 41])
            if save == True:
                joined_path = os.path.join(save_path, 'abalone_train_preds_vs_actual.png')
                plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            sub_df = preds_df.loc[preds_df['data'] == 'test']
            for j, t in enumerate(thresh):
                thresh_df = sub_df.loc[sub_df.gmean_metric >= t]
                rmse = root_mean_squared_error(thresh_df.Rings, thresh_df.preds)
                ax.scatter(thresh_df.preds, thresh_df.Rings, s=40, label=f'{round(t, 2)} (RMSE = {rmse:.2f})', color=colors[j], alpha=0.3)
            ax.set_xlabel('Predicted Rings', fontsize=13)
            ax.set_ylabel('Actual Rings', fontsize=13)
            ax.legend(title='Threshold Level', fontsize=11, title_fontsize=11, loc='lower right')
            ax.tick_params(axis='both', which='major', labelsize=11)
            ax.plot([41,0], [41,0], color='black', ls=':', alpha=0.75)
            ax.set_xlim([-1, 41])
            ax.set_ylim([-1, 41])
            if save == True:
                joined_path = os.path.join(save_path, 'abalone_test_preds_vs_actual.png')
                plt.savefig(joined_path, format='png', dpi=300, bbox_inches='tight')
            plt.close()