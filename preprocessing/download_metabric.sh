### DOWNLOAD AND PROCESS RAW DATA FILES, SAVE FINAL TSVs TO DATA FOLDER
filepath=${1} # path to cloned sage repository

cd ${filepath}/data/raw_data
mkdir metabric
cd metabric
# Download METABRIC gene expression data
wget  https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz
tar -xf brca_metabric.tar.gz brca_metabric/data_mrna_illumina_microarray.txt
rm brca_metabric.tar.gz
# Download METABRIC metadata
curl --remote-name --remote-header-name 'https://api.gdc.cancer.gov/data/6b22fbef-fbdc-46e3-adf0-cc341c929a59'

cd ${filepath}/gdan-tmp-models

# Convert METABRIC data file gene names
python tools/convert.py \
	--data ${filepath}/data/raw_data/metabric/brca_metabric/data_mrna_illumina_microarray.txt \
	--out ${filepath}/data/raw_data/metabric/brca_metabric/metabric_converted.tsv \
	--cancer BRCA \
	--conversion_file tools/ft_name_convert/entrez2tmp_BRCA_GEXP.json \
	--delete_i_col 1

# Quantile Rescale
bash tools/run_transform.sh \
  ${filepath}/data/raw_data/metabric/brca_metabric/metabric_converted.tsv \
    BRCA

# Handle 10 quantile differences
python tools/zero_floor.py \
  -in user-transformed-data/transformed-data.tsv \
  -out user-transformed-data/transformed-data.tsv

# rename file and move to raw_data folder
mv user-transformed-data/transformed-data.tsv ${filepath}/data/raw_data/metabric/brca_metabric/metabric_rescaled_floored.tsv