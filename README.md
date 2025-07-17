# Supervised Autoencoder for Generalization Estimates (SAGE)
<p align="center">
	<img align="center" width="585" height="300" alt="fig1(7)" src="https://github.com/user-attachments/assets/931741b9-7fd0-4b44-b659-14e8c38583fe" />
</p>

## Description
Failure of machine learning models to generalize to new data is a core problem limiting their reliability, 
partly due to the lack of simple and robust methods for comparing new data to the original training dataset. 
We propose a standardized approach for assessing data similarity with a supervised autoencoder for generalization estimates (SAGE). 
Here, we train a SAGE model on the popular HAM10000 dermoscopic imaging dataset and use it to probe similarity with skin lesion images from three other academic hospitals. 
SAGE can be used to uncover problematic image artefacts and to improve performance of a separate malignancy predictor, as we show in our paper 
<ins>_Ensemble out-of-distribution detection improves skin cancer malignancy prediction_</ins>.

## Setup
### 1. Datasets
Download these publicly-available imaging datasets:
* [Humans Against Machine (HAM) 100000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T), Tschandl et al. 2018
* [Hospital Italiano de Buenos Aires (HIBA)](https://api.isic-archive.com/doi/hospital-italiano-de-buenos-aires-skin-lesions-images-2019-2022/), Ricci Lara et al. 2023
* [Universidade Federal do Espírito Santo (UFES)](https://data.mendeley.com/datasets/zr7vgbcyr2/1), Pacheco et al. 2020
* [Diverse Dermatology Images (DDI)](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965), Daneshjou et al. 2022

### 2. Organization
Structure your image folders as follows:
```
data_directory/
├── dataset_name/
│   ├── images/
│   └── metadata.csv
```
Each dataset must have its own folder with an `images` subdirectory and an associated `metadata` CSV file.

### 3. Build the Environment
The python environment used to run this code is built with `conda` and uses `pip` for package installs.
First, create the environment using the provided YAML file.
```
conda env create -f sage.yml
```
Then, activate your environment.
```
conda activate sage
```

## Training
Replace the filepaths in the following script to train your own SAGE model on the HAM10000 dataset.
```
python3 train_ham.py \
--imagedir /path/to/data_directory/ham/images \
--metafile /path/to/data_directory/ham/metadata.csv \
--savedir /path/to/model/savedir \
--encoder ResNet \ # default
--dim 32 # default
```
We provide support for three options of pre-trained encoders: `ResNet`, `Inception` and `ViT`. Our trained ResNet50 model is also available for download as
a zip file from [this link](https://drive.google.com/drive/folders/1wcMIaFtooOJuJ1h3Ct_VcfNC0yorGc33?usp=sharing).

## Scoring
### HAM vs. HIBA, UFES and DDI
If you've downloaded the datasets as shown above, replace the filepaths in the following script to calculate SAGE scores.
```
python3 score_all.py \
--encoder ResNet \ # match the encoder type of your trained model
--dim 32 \ # match latent space size of your trained model
--modelpth /path/to/trained/model \
--datadir /path/to/data_directory # dir with imaging datasets
--outdir /path/to/scores/outdir \
```
This will output `pickle` files of `pandas` dataframe objects for 1) SAGE model outputs and 2) the score values associated with each image.

### HAM vs. Your Data
You can score a separate skin lesion imaging dataset of your choosing against HAM10000 so long as the directory structure is the same as described above. 
Again, replace the filepaths and include your dataset's name after the `--compare-to` arg to use the following script.
```
python3 score_other.py
--encoder ResNet \ # match the encoder type of your trained model
--dim 32 \ # match latent space size of your trained model
--modelpth /path/to/trained/model \
--datadir /path/to/data_directory # dir with imaging datasets
--compare-to dataset_name 
--outdir /path/to/scores/outdir \
```
