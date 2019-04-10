from gensim import  models
from gensim.corpora import Dictionary
from gensim.matutils import  cossim
import pandas as pd
from numpy import linalg


"""
local_letter: str
Term
frequency
weighing, one
of:
*`n` - natural,
*`l` - logarithm,
*`a` - augmented,
*`b` - boolean,
*`L` - log
average.
global_letter: str
Document
frequency
weighting, one
of:
*`n` - none,
*`t` - idf,
*`p` - prob
idf.
normalization_letter: str
Document
normalization, one
of:
*`n` - none,
*`c` - cosine.
"""

def get_all_irs():
    local_frequencies = ['n','l','a','b','L']
    weighting = ['n','t','p']
    doc_normalization = ['n','c']

    all_combinations = [ x+y+z for x in local_frequencies for y in weighting for z in doc_normalization]
    return all_combinations


def create_tf_idf_model(df,smartirs):
    dct =  Dictionary([str(x).split() for x in df['question1'].tolist()+df['question2'].tolist() if (type(x) is not float) and (type(x) is not int) ])
    corpus = [dct.doc2bow(line.split()) for line in df.question1.tolist()+df.question2.tolist()  if  (type(line) is not float) and (type(line) is not int) ]
    model = models.TfidfModel(corpus,smartirs=smartirs) ## model
    return model,dct

"""
 Normalize the vector.
"""
def normalize_vecotr(vector):
    norm = linalg.norm(vector)
    return vector/norm



"""
similarity function
"""

def calculate_similarity_single(model,dictionary,question1,question2):
    if type(question1) is float:
        print(question1)
        return 1
    q1_d2b =  dictionary.doc2bow([x for x in question1.split() if  type(x) is not float  and type(x) is not int ])
    q2_d2b = dictionary.doc2bow([x for x in question2.split() if  type(x) is not float  and type(x) is not int ])
    #print(q2_d2b)
    q1_vec = model[q1_d2b]
    q2_vec = model[q2_d2b]
    #print(q1_vec)
    cosine_value = cossim(q1_vec,q2_vec)
    return cosine_value

"""
Calculate the the similarity for all the dataframe.
"""
def calculate_similarity(model,df,dictionary):
    sim_score = []
    for index,row in df.iterrows():
        sim_val = calculate_similarity_single(model,dictionary,row.question1,row.question2)
        #print(sim_val)
        sim_score.append(sim_val)

    return sim_score
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
 To lower case is applied to DF.
"""
LOWER_DF = True

import os
def compute_all_similarities_train_and_test(train_df,test_df,data_dir='../data',file_suffix='idf_appended'):
    train_df.fillna("",inplace=True)
    test_df.fillna("", inplace=True)
    if LOWER_DF:
        lower_all_text(train_df)
        lower_all_text(test_df)
        print("To Lower DF completed")
    all_smartirs = get_all_irs()
    for irs in all_smartirs:
        irs_model,dictionary = create_tf_idf_model(train_df,irs)
        sim_scores = calculate_similarity(irs_model,train_df,dictionary)
        train_df['tfidf_'+irs] = sim_scores
        sim_scores = calculate_similarity(irs_model,test_df,dictionary)
        test_df['tfidf_'+irs] = sim_scores
        print('Finished {}'.format(irs))
    file_name = os.path.join(data_dir,'test_'+file_suffix)
    test_df.to_csv(file_name,index=False)
    file_name = os.path.join(data_dir,'train_'+file_suffix)
    train_df.to_csv(file_name,index=False)



def test():
    #df = pd.read_csv('../data/test.csv',nrows=200000)
    #model,dictionary = create_tf_idf_model(df)
    #sims = calculate_similarity(model,df,dictionary)
    #print([ x for x in sims if x!=1.0])
    test_df = pd.read_csv('../data/test_processed.csv',nrows=200000)
    train_df = test_df = pd.read_csv('../data/train_processed.csv',nrows=200000)
    compute_all_similarities_train_and_test(train_df,test_df)


if __name__ == '__main__':
    test()





