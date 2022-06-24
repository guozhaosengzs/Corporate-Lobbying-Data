import pandas as pd 
import os.path
from classification import KeyBERTEventExtractor, CrossEncoderEventExtractor
from pprint import pprint

def main():

    list_of_files = ['Colorado_bills_summaries.csv',
        'Iowa_bills_summaries.csv',
        'Nebraska_bills_summaries.csv',
        'Wisconsin_bills_summaries.csv']
    
    for f in list_of_files:
        print('Working on', f)

        classification(f)

def classification(file_name):

    df = pd.read_csv(file_name)
    print(df.head())
    return 
    variable_list = ['CAP_label' + str(i) for i in range(1,6)] + ['CAP_score' + str(i) for i in range(1,6)] + ['MSCI_label' + str(i) for i in range(1,6)] + ['MSCI_score' + str(i) for i in range(1,6)] 

    for col in variable_list:
        df[col] = ""

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
    # extractor = CrossEncoderEventExtractor()

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

        df_classification.at[i,'CAP_label1'] = cat_dict[labels[0]]
        df_classification.at[i,'CAP_label2'] = cat_dict[labels[1]]
        df_classification.at[i,'CAP_label3'] = cat_dict[labels[2]]
        df_classification.at[i,'CAP_label4'] = cat_dict[labels[3]]
        df_classification.at[i,'CAP_label5'] = cat_dict[labels[4]]

        df_classification.to_csv('assy_88_classify_first50_top5.csv', index=False)

if __name__ == '__main__':
    main()