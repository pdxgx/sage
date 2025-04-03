import pandas as pd
import re
import json

# make label conversion dictionaries
label_dict_meta = {
    'LumA':'BRCA_1',
    'LumB':'BRCA_2',
    'Basal':'BRCA_3',
    'Her2':'BRCA_4',
    'Normal':'BRCA_5',
    'Claudin':'BRCA_6'
}
label_dict_tcga = {'BRCA_1':0, 'BRCA_2':1, 'BRCA_3':2, 'BRCA_4':3, 'BRCA_5':4, 'BRCA_6':5}

# only ['skgrid', 'aklimate', 'subscope'] models have GEXP features shared across TCGA, AURORA and METABRIC
with open('models/model_info.json', 'r') as fh:
    data = json.load(fh)
    ft_list = []
    # add all eligible model features
    for mod in ['skgrid', 'aklimate', 'subscope']:
        ft_list.extend(data[mod]['BRCA']['GEXP']['fts'])
    ft_set = sorted(list(set(ft_list))) # removes duplicate features, sort ensures consistent order

# read TCGA data
tcga_file = 'gdan-tmp-models/tools/TMP_20230209/BRCA_v12_20210228.tsv'
cols = pd.read_csv(tcga_file, sep='\t', header=0, nrows=1).columns
# columns to keep: Accession, Subtype label, Selected GEXP features
usecols = cols[:2].to_list() + ft_set
tcga = pd.read_csv(tcga_file, sep='\t', header=0, usecols=usecols)
tcga.to_csv('data/TCGA_BRCA.tsv', sep='\t', index=False)

# prep file for tmp model calls
gexp_cols = [c for c in tcga.columns if 'N:GEXP' in c]
tcga_slim = tcga[['BRCA'] + gexp_cols]

# read METABRIC data and save
m_file = 'data/raw_data/metabric/brca_metabric/metabric_rescaled_floored.tsv'
usecols = ['Unnamed: 0'] + ft_set
metabric = pd.read_csv(m_file, sep='\t', header=0, usecols=usecols)

m_labs_file = 'data/raw_data/metabric/METABRIC_PAM50_silhouettes_no_Normal-Like_2022-03-15.tsv'
m_labs = pd.read_csv(m_labs_file, sep='\t')
m_labs['Subtype'] = m_labs['Subtype'].replace(label_dict_meta)

metabric = pd.merge(metabric, m_labs, on='Unnamed: 0') # skips all unlabeled rows ~1,000
metabric = metabric.rename({'Unnamed: 0': 'BRCA', 'Subtype': 'Labels'}, axis='columns')
metabric.to_csv('data/METABRIC_BRCA.tsv', sep='\t', index=False)

gexp_cols = [c for c in metabric.columns if 'N:GEXP' in c]
met_slim = metabric[['BRCA'] + gexp_cols]

def format_aurora_id(x):
    match = re.search(r"^(.*?)(?=-R-)", x)
    return match.group(0)

def format_bcr(x):
    y = x.replace('.', '-')
    return y

a_file = 'data/raw_data/aurora/aurora_rescaled_floored.tsv'
aurora = pd.read_csv(a_file, sep='\t')
aurora = aurora[usecols]
aurora['Unnamed: 0'] = aurora['Unnamed: 0'].apply(format_aurora_id)
a_labs_file = 'data/raw_data/aurora/Supplementary_Table.2.xlsx'
a_labs = pd.read_excel(a_labs_file, sheet_name='2.AURORA study')
a_labs['Unnamed: 0'] = a_labs['BCR Portion barcode'].apply(format_bcr)
a_labs['PAM50 Call'] = a_labs['PAM50 Call'].replace(label_dict_meta)

# Unnamed: 0, Tissue type, Sample Type, PAM50 Call
a_labs = a_labs[['Unnamed: 0', 'PAM50 Call', 'Tissue type', 'Sample Type', 'Race']]
aurora = pd.merge(aurora, a_labs, on='Unnamed: 0')
aurora = aurora.rename({'Unnamed: 0': 'BRCA', 'PAM50 Call': 'Labels'}, axis='columns')
aurora.to_csv('data/AURORA_BRCA.tsv', sep='\t', index=False)

gexp_cols = [c for c in aurora.columns if 'N:GEXP' in c]
aur_slim = aurora[['BRCA'] + gexp_cols]

# combine dataframes from each dataset
combined_slim = pd.concat([tcga_slim, met_slim, aur_slim])
combined_slim = combined_slim.rename(columns={'BRCA':''}) # blank column for samples
combined_slim.to_csv('gdan-tmp-models/user-transformed-data/TCGA_METABRIC_AURORA_BRCA_tmp.tsv')