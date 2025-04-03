import pandas as pd

a_file = 'GSE209998_AUR_129_raw_counts_TMP.BRCA.Features.tsv'
aurora = pd.read_csv(a_file, sep='\t').T
aurora = aurora.reset_index()
aurora.columns = aurora.iloc[0]
aurora = aurora.drop(0).reset_index(drop=True)

# save transposed file
aurora.to_csv('aurora_transposed.tsv', sep='\t', index=False)