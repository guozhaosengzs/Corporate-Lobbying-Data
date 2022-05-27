import pandas as pd

df = pd.read_csv("Iowa_allyrs_WRDS_empath_npscore.csv")

df_small = df[['Bill', 'assembly_number']]

df_small.to_csv('bill_assm.csv', index=False)
