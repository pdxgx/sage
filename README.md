# Supervised Autoencoders for Generalization Estimates (SAGE)

<p align="center">
<img width="585" height="300" alt="fig2" src="https://github.com/user-attachments/assets/3706c5db-6bbd-43dd-bd98-8b4f44beb338" />
</p>

## Description
Failure of machine learning models to generalize to new data is a core problem limiting their reliability, 
partly due to the lack of simple and robust methods for comparing new data to a model's original training dataset. 
We propose a standardized approach for assessing similarity between datasets with supervised autoencoders for generalization estimates (SAGE).

Here, we train a SAGE model ensemble on the popular HAM10000 dermoscopic imaging dataset and use it to quantify uncertainty of skin lesion images from  North America, South America, Europe and Australia. 
SAGE can be used to uncover problematic image artifacts and gate a downstream classifier by identifying samples under distribution shift. 

Our preprint can be accessed [here](https://www.medrxiv.org/content/10.1101/2025.08.20.25334101v2) and the paper, ***Multi-criterion uncertainty estimation improves skin cancer distribution shift detection and malignancy prediction***, will be published in _npj Digital Medicine_.

## Setup
### 1. Datasets
Download these publicly-available imaging datasets:
* [Humans Against Machine (HAM) 100000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T), Tschandl et al. 2018
* [Hospital Italiano de Buenos Aires (HIBA)](https://api.isic-archive.com/doi/hospital-italiano-de-buenos-aires-skin-lesions-images-2019-2022/), Ricci Lara et al. 2023
* [Universidade Federal do Espírito Santo (UFES)](https://data.mendeley.com/datasets/zr7vgbcyr2/1), Pacheco et al. 2020
* [Diverse Dermatology Images (DDI)](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965), Daneshjou et al. 2022

### 2. Organization
Structure your image dataset directory:
```
data/
├── dataset_name/
│   ├── images/
│   └── metadata.csv
```
Each dataset must have its own folder with an `images` subdirectory and an associated `metadata` CSV file.

### 3. Download Dependencies
We recommend building a virtual environment with `venv` or `conda`. You can install requirements within your environment using `pip` as follows:
```
pip install -r requirements.txt
```

## Training
Enter the `scripts` subdirectory and replace the filepaths in the following command to train your own SAGE ensemble on HAM10000. This will train and save checkpoints for 5 SAGE models.
```
python3 train_ham.py \
--imagedir /path/to/main_directory/ham/images \
--metafile /path/to/main_directory/ham/metadata.csv \
--savedir /path/to/model/savedir \
--encoder ResNet \
--dim 256 # default
```
We provide support for CUDA and three options of pre-trained encoders: `ResNet`, `Inception` and `ViT`. 
Default settings will train SAGE with a latent embedding size of 256. Pre-trained ensemble weights for SAGE with a ResNet encoder are available upon request.

## Scoring
### HAM vs. [HIBA, UFES, DDI, MILK10K, Caltech 101]
If you've downloaded the datasets as shown above, replace the filepaths in the following command to calculate SAGE scores.
```
python3 sage_score_all.py \
--encoder ResNet \
--dim 256 \
--modelpth /path/to/trained/models \
--datadir /path/to/main_directory \
--outdir /path/to/scores/outdir
```
This will output `pickle` files of `pandas` dataframe objects for 1) SAGE model outputs and 2) the score values associated with each image.

### HAM vs. Your Data
You can score a separate imaging dataset of your choosing against HAM10000 so long as the directory structure follows the specified
[organization](#2-organization).
Your dataset folder must contain a `metadata` CSV file with at minimum an `img_id` column for unique identifiers corresponding to image filenames and a `label` column.
If diagnostic labels are unknown or irrelevant, simply fill the `label` column with NA values or any integer.
Use the following command after replacing filepaths and include the eval dataset name after the `--compare-to` argument.
```
python3 sage_score_other.py \
--encoder ResNet \
--dim 256 \
--modelpth /path/to/trained/models \
--datadir /path/to/main_directory \
--compare-to dataset_name \
--outdir /path/to/scores/outdir
```

