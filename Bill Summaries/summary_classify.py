import pandas as pd 
import numpy as np
import os.path
from classification import KeyBERTEventExtractor, CrossEncoderEventExtractor

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
    df['summary'] = df['summary'].astype(str)

    variable_list = ['CAP_label' + str(i) for i in range(1,6)] + ['CAP_score' + str(i) for i in range(1,6)] + ['MSCI_label' + str(i) for i in range(1,6)] + ['MSCI_score' + str(i) for i in range(1,6)] 

    for col in variable_list:
        df[col] = ""

    CAP_df = pd.read_csv('category_CAP.csv')
    CAP_df['full_text'] = CAP_df['Subtopic'] + ' ' + CAP_df['Topic'] + ', ' + CAP_df['Description'] 
    CAP_dict = dict(zip(CAP_df.full_text, CAP_df.Code))
    CAP_categories = list(CAP_dict.keys())

    MSCI_df = pd.read_csv('category_MSCI.csv')
    MSCI_dict = dict(zip(MSCI_df.Description, MSCI_df.Code))
    MSCI_categories = list(MSCI_dict.keys())    

    extractor = KeyBERTEventExtractor()
    # extractor = CrossEncoderEventExtractor() ## Use this line for better but 100X times slower results

    for i, row in df.iterrows():
        summary = row.summary

        if summary=='nan':
            continue

        CAP_output = extractor.extract([summary], CAP_categories)[0]
        CAP_labels = CAP_output['events']
        CAP_scores = CAP_output['scores']

        for j, col in enumerate(variable_list[:5]):
            df.at[i, col] = CAP_dict[CAP_labels[j]]

        for j, col in enumerate(variable_list[5:10]):
            df.at[i, col] = CAP_scores[j]


        MSCI_output = extractor.extract([summary], MSCI_categories)[0]
        MSCI_labels = MSCI_output['events']
        MSCI_scores = MSCI_output['scores']

        for k, col in enumerate(variable_list[10:15]):
            df.at[i, col] = MSCI_dict[MSCI_labels[k]]

        for k, col in enumerate(variable_list[15:20]):
            df.at[i, col] = MSCI_scores[k]

    df.to_csv('finished_' + file_name, index=False)

if __name__ == '__main__':
    main()