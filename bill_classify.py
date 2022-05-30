import pandas as pd 
import os.path
from classification import CrossEncoderEventExtractor
from pprint import pprint

def main():
    records_df = pd.read_csv('bill_assm.csv')

    df_classification = records_df.copy()
    df_classification.drop_duplicates(inplace=True, ignore_index=True)

    df_classification['category'] = ""

    df_classification = df_classification.query("assembly_number == 88")

    category_df = pd.read_csv('category.csv')
    category_df['full_text'] = category_df['Subtopic'] + ' ' + category_df['Topic'] + ', ' + category_df['Description'] 
    # print(category_df['full_text'])

    cat_dict = dict(zip(category_df.full_text, category_df.Code))
    categories = list(cat_dict.keys())

    for i, row in df_classification.iterrows():
        bill = str(row.Bill).replace(' ', '')
        assy = str(row.assembly_number)

        file_path = os.path.join('Iowa', 'bills_text_' + assy, 'iowa_general_assembly_' + assy + '_' + bill + ".txt")
        
        try:
            with open(file_path, encoding="utf8") as f:
                bill_text = f.read().replace('\n', ' ')
                print(file_path)
        except:
            print("file not found:", file_path)
            continue

        extractor = CrossEncoderEventExtractor()
        label = extractor.extract([bill_text], categories)[0]['events'][0]


        df_classification.at[i,'category'] = cat_dict[label]
        df_classification.to_csv('assy_88_classify.csv', index=False)

if __name__ == '__main__':
    main()