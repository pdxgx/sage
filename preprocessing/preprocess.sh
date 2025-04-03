# run shell script as "bash preprocessing/preprocess.sh {sage_directory_path}" from sage_directory

filepath=${1} # path to cloned sage repository

cd ${filepath}/gdan-tmp-models/tools
# download source matrix
curl --remote-name --remote-header-name \
'https://api.gdc.cancer.gov/data/5116e86f-7646-4b7b-9d6e-dafddf2cc0f3'
tar -xzf TMP_20230209.tar.gz
rm TMP_20230209.tar.gz
cd TMP_20230209
# Loop through all .tsv files in the directory
for file in *.tsv; do
  # If the filename does not contain 'BRCA', remove it
  if [[ ! "$file" =~ BRCA ]]; then
    rm "$file"
  fi
done
cd ${filepath}/gdan-tmp-models/tools
# download gene conversion files
curl --remote-name --remote-header-name \
'https://api.gdc.cancer.gov/data/01482f73-0fd8-4bef-a9d7-63ecbe37b75a'
tar -xf ft_name_convert.tar.gz
rm ft_name_convert.tar.gz
cd ft_name_convert
# Loop through all .json files in the directory
for file in *.json; do
  # If the filename does not contain 'BRCA', remove it
  if [[ ! "$file" =~ BRCA ]]; then
    rm "$file"
  fi
done
cd ${filepath}/models
# download model info json
curl --remote-name --remote-header-name \
'https://api.gdc.cancer.gov/data/94a477f5-947b-4412-86b0-591aec7d6191'

cd ${filepath}
bash preprocessing/download_metabric.sh ${filepath}
bash preprocessing/download_aurora.sh ${filepath}
python preprocessing/format_matrices.py

#bash RUN_MODEL.sh BRCA GEXP aklimate user-transformed-data/TCGA_BRCA_slim.tsv | \
#bash RUN_MODEL.sh BRCA GEXP subscope user-transformed-data/TCGA_BRCA_slim.tsv | \
#bash RUN_MODEL.sh BRCA GEXP skgrid user-transformed-data/TCGA_BRCA_slim.tsv | \
#bash RUN_MODEL.sh BRCA GEXP aklimate user-transformed-data/METABRIC_BRCA_slim.tsv | \
#bash RUN_MODEL.sh BRCA GEXP subscope user-transformed-data/METABRIC_BRCA_slim.tsv | \
#bash RUN_MODEL.sh BRCA GEXP skgrid user-transformed-data/METABRIC_BRCA_slim.tsv | \
#bash RUN_MODEL.sh BRCA GEXP aklimate user-transformed-data/AURORA_BRCA_slim.tsv | \
#bash RUN_MODEL.sh BRCA GEXP subscope user-transformed-data/AURORA_BRCA_slim.tsv | \
#bash RUN_MODEL.sh BRCA GEXP skgrid user-transformed-data/AURORA_BRCA_slim.tsv