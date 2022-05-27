import pandas as pd 

records_df = pd.read_csv('bill_assm.csv')

df_classification = records_df.copy()
df_classification.drop_duplicates(inplace=True, ignore_index=True)

df_classification.to_csv('test.csv', index=False)

"iowa_general_assembly_83_HSB1.txt"