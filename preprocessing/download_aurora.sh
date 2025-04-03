filepath=${1} # path to cloned sage repository

cd ${filepath}/data/raw_data/aurora

# to download raw, unprocessed reads from AURORA paper ! note gene names are not in HUGO format and will result in an error when using convert.py
#wget 'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209998/suppl/GSE209998_AUR_129_raw_counts.txt.gz'
#gunzip GSE209998_AUR_129_raw_counts.txt.gz
# assign Entrez Gene IDs
# convert AURORA gene names to TMP format?
#python tools/convert.py \
	#--data ${filepath}/data/raw_data/aurora/GSE209998_AUR_129_raw_counts_renamed.tsv \
	#--out ${filepath}/data/raw_data/aurora/aurora_BRCA_GEXP.tsv \
	#--cancer BRCA
    
python ${filepath}/preprocessing/aurora_prepare.py

cd ${filepath}/gdan-tmp-models

# Quantile Rescale
bash tools/run_transform.sh \
  ${filepath}/data/raw_data/aurora/aurora_transposed.tsv \
    BRCA

# Handle 10 quantile differences
python tools/zero_floor.py \
  -in user-transformed-data/transformed-data.tsv \
  -out user-transformed-data/transformed-data.tsv

# rename file and move to raw_data folder
mv user-transformed-data/transformed-data.tsv ${filepath}/data/raw_data/aurora/aurora_rescaled_floored.tsv