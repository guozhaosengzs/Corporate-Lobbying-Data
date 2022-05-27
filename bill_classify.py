import pandas as pd 
import os.path

records_df = pd.read_csv('bill_assm.csv')

df_classification = records_df.copy()
df_classification.drop_duplicates(inplace=True, ignore_index=True)

df_classification.to_csv('test.csv', index=False)
df_classification['category'] = ""

df_classification = df_classification.query("assembly_number == 88")

for i, row in df_classification.iterrows():
    bill = str(row.Bill).replace(' ', '')
    assy = str(row.assembly_number)

    file_path = os.path.join('Iowa', 'bills_text_' + assy, 'iowa_general_assembly_' + assy + '_' + bill + ".txt")
    print(file_path)

    with open(file_path, encoding="utf8") as f:
        contents = f.read().replace('\n', '')

    print(contents)


    df_classification.at[i,'category'] = 2512323
    
    break
# print(df_classification)

"iowa_general_assembly_83_HSB1.txt"