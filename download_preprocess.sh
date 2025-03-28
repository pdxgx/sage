# run shell script as "bash tools/download_preprocess.sh {sage_directory_path}"
sage_path=${1} # path to sage directory

cd sage_path/tools
# download gene conversion files
curl --remote-name --remote-header-name \
'https://api.gdc.cancer.gov/data/01482f73-0fd8-4bef-a9d7-63ecbe37b75a'
cd sage_path/tools/ft_name_convert
# Loop through all .json files in the directory
for file in *.json; do
  # If the filename does not contain 'BRCA', remove it
  if [[ ! "$file" =~ BRCA ]]; then
    rm "$file"
  fi
done

cd sage_path/data/raw_data/tcga
# Download processed TCGA data matrix
curl --remote-name --remote-header-name \
'https://api.gdc.cancer.gov/data/5116e86f-7646-4b7b-9d6e-dafddf2cc0f3'
cd sage_path/data/raw_data/tcga/TMP_20230209
# Loop through all .tsv files in the directory
for file in *.tsv; do
  # If the filename does not contain 'BRCA', remove it
  if [[ ! "$file" =~ BRCA ]]; then
    rm "$file"
  fi
done

# Download METABRIC gene expression data
cd sage_path/data/raw_data/metabric
wget  https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz
tar -xf brca_metabric.tar.gz brca_metabric/data_mrna_illumina_microarray.txt
rm brca_metabric.tar.gz

# Download METABRIC metadata
curl --remote-name --remote-header-name \
'https://api.gdc.cancer.gov/data/6b22fbef-fbdc-46e3-adf0-cc341c929a59'

# Convert METABRIC data file gene names
cd sage_path
python tools/convert.py \
	--data data/raw_data/metabric/brca_metabric/data_mrna_illumina_microarray.txt \
	--out data/raw_data/metabric/brca_metabric/cbioportal_BRCA_GEXP.tsv \
	--cancer BRCA \
	--conversion_file tools/ft_name_convert/entrez2tmp_BRCA_GEXP.json \
	--delete_i_col 1

# raw AURORA data downloaded from the following link:
#https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE209998
python tools/aurora_transpose.py

# Quantile Rescale METABRIC and AURORA files
bash tools/run_transform.sh \
  data/raw_data/metabric/brca_metabric/cbioportal_BRCA_GEXP.tsv \
  data/raw_data/metabric/brca_metabric/metabric_rescaled.tsv

bash tools/run_transform.sh \
  data/raw_data/aurora/AUR_129_TMP_BRCA_transposed.tsv \
  data/raw_data/aurora/aurora_rescaled.tsv

# Handle 10 quantile differences
python tools/zero_floor.py \
  -in data/raw_data/metabric/brca_metabric/metabric_rescaled.tsv \
  -out data/raw_data/metabric/brca_metabric/metabric_rescaled_floored.tsv

python tools/zero_floor.py \
  -in data/raw_data/aurora/aurora_rescaled.tsv \
  -out data/raw_data/aurora/aurora_rescaled_floored.tsv

python tools/format_matrices.py