from model_setup import *
import re
import glob

files = glob.glob('./TMP_20230209/*.tsv')
files = [f for f in files if 'folds' not in f] # exclude CV splits
for fn in files:
    cols = pd.read_csv(fn, sep='\t', header=0, nrows=1).columns
    # columns to keep: Accession, Subtype label, Gene expression values
    usecols = cols[:2].to_list() + [c for c in cols if c.startswith('N:GEXP')]
    df = pd.read_csv(fn, sep='\t', header=0, usecols=usecols)
    ids = []
    for col in usecols[2:]:
        match = re.search(r':(\d+):$', col) # only match NCBI GeneID
        # to get NCBI gene name: match = re.search(r'::([^:]+):', col)
        if match:
            ids.append(match.group(1))
        else:
            raise AttributeError(f'MATCH FAILED! Column: {col}')
    df.columns = cols[:2].to_list() + ids

    # reassign labels
    labs = []
    cans = []
    for la in df['Labels'].to_list():
        # match cancer type abbreviation
        match = re.search(r'^(.+)_', la)  # matches at beginning of label
        if match:
            cans.append(match.group(1))
        else:
            raise AttributeError(f'MATCH FAILED! Label: {la}')
        # match subtype number
        match = re.search(r'_(\d+)$', la)
        if match:
            labs.append(int(match.group(1))) # convert to int
        else:
            raise AttributeError(f'MATCH FAILED! Label: {la}')
    df['Cancer'] = cans
    df['Labels'] = labs

    # Now we have a cleaned df of the gene expression data with appropriate labels.
    # We can add to one large df for training/fine-tuning our SAGE models in a leave-
    # one-out fashion.

