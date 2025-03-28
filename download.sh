# Download Gene Expression Data
cd ~/scratch/sage/data || exit
mkdir metabric
cd metabric || exit
wget  https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz
tar -xf brca_metabric.tar.gz brca_metabric/data_mrna_illumina_microarray.txt
rm brca_metabric.tar.gz

python tools/convert.py \
	--data brca_metabric/data_mrna_illumina_microarray.txt \
	--out cbioportal_BRCA_GEXP.tsv \
	--cancer BRCA \
	--conversion_file tools/ft_name_convert/entrez2tmp_BRCA_GEXP.json \
	--delete_i_col 1