import pandas as pd 
import os.path
from classification import KeyBERTEventExtractor, CrossEncoderEventExtractor
from pprint import pprint

def main():
    records_df = pd.read_csv('bill_assm.csv')

    df_classification = records_df.copy()
    df_classification.drop_duplicates(inplace=True, ignore_index=True)

    df_classification['category1'] = ""
    df_classification['category2'] = ""
    df_classification['category3'] = ""
    df_classification['category4'] = ""
    df_classification['category5'] = ""

    df_classification = df_classification.query("assembly_number == 88").head(50)

    category_df = pd.read_csv('category.csv')
    category_df['full_text'] = category_df['Subtopic'] + ' ' + category_df['Topic'] + ', ' + category_df['Description'] 
    # print(category_df['full_text'])

    cat_dict = dict(zip(category_df.full_text, category_df.Code))
    categories = list(cat_dict.keys())

    extractor = KeyBERTEventExtractor()
    extractor = CrossEncoderEventExtractor()

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
        
        labels = extractor.extract([bill_text], categories)[0]['events']

        df_classification.at[i,'category1'] = cat_dict[labels[0]]
        df_classification.at[i,'category2'] = cat_dict[labels[1]]
        df_classification.at[i,'category3'] = cat_dict[labels[2]]
        df_classification.at[i,'category4'] = cat_dict[labels[3]]
        df_classification.at[i,'category5'] = cat_dict[labels[4]]

        df_classification.to_csv('assy_88_classify_first50_top5.csv', index=False)

if __name__ == '__main__':
    main()