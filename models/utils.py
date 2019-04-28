from gensim import corpora
import os
import pandas as pd

"""
Helper function to lower string.
"""
def lower_item(item):
    return item.lower()
"""
Lower text1 and text2
"""
def lower_all_text(df):
    df.question1 = df.question1.map(lower_item)
    df.question2 = df.question2.map(lower_item)


"""
Helper to get all the text. All the preprocessing should be done here. Including nltk stemming/lemmatization.
"""
def get_text_array(df):
    return [ x.split() for x in df.question1.tolist()+df.question2.tolist()]



def get_processed_df(data_dir='../data',file_suffix='processed.csv'):
    test_fp = os.path.join(data_dir,'test_'+file_suffix)
    train_fp = os.path.join(data_dir,'train_'+file_suffix)
    train_df = pd.read_csv(train_fp)
    # Drop NA values
    train_df = train_df.dropna()
    test_df = pd.read_csv(test_fp)
    test_df = test_df.dropna()
    return train_df,test_df

