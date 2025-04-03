# run script as "bash preprocessing/predict.sh {sage_directory_path}" from sage_directory
# call pre-trained TMP models ! note must be logged into Synapse and have Docker daemon running
filepath=${1} # path to cloned sage repository
cd ${filepath}/gdan-tmp-models
bash RUN_MODEL.sh BRCA GEXP aklimate user-transformed-data/TCGA_METABRIC_AURORA_BRCA_tmp.tsv
bash RUN_MODEL.sh BRCA GEXP subscope user-transformed-data/TCGA_METABRIC_AURORA_BRCA_tmp.tsv
bash RUN_MODEL.sh BRCA GEXP skgrid user-transformed-data/TCGA_METABRIC_AURORA_BRCA_tmp.tsv
# migrate results
bash tools/migrate.sh BRCA GEXP ${filepath}/data/tmp_preds
# build results file
bash tools/build_results_file.sh tmp_preds.tsv ${filepath}/data/tmp_preds BRCA GEXP aklimate.subscope.skgrid