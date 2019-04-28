

from gensim import  corpora
from gensim.models import  LdaModel
import  utils as ut
import  pickle
import  pandas as pd
import  os
NUM_TOPICS = 50

def build_topic(df,load_existing=True):

    words = ut.get_text_array(df)
    dictionary = corpora.Dictionary(words)
    if load_existing and os.path.exists('lda_model.h5'):
        model = LdaModel.load('lda_model.h5')
        return model,dictionary


    corpus = [dictionary.doc2bow(text) for text in words]
    model = LdaModel(corpus,num_topics=NUM_TOPICS)
    model.save('lda_model.h5')
    return model,dictionary


def create_topic_headers():
    headers = ['q1_topic_'+str(x) for x in range(NUM_TOPICS)] + ['q2_topic_'+str(x) for x in range(NUM_TOPICS)]
    return headers

def remap_the_topics(vector):
    result = [0 for x in range(NUM_TOPICS)]
    for x in vector:
        result[x[0]] = x[1]
    return result

def get_topic_q1_and_q2(model,dictionary,row):
    q_1 =  dictionary.doc2bow(row.question1.split())
    q_2 = dictionary.doc2bow(row.question2.split())
    result = remap_the_topics(model.get_document_topics(q_1))+ remap_the_topics(model.get_document_topics(q_2))
    return result


"""
x[1] is topic score
"""
def process_topic(q1_topic,q2_topic):
    return [x[1] for  x in q1_topic] + [x[1] for  x in q2_topic]

import os
def build_topics_scores(train_df,test_df,inplace=True,data_folder ='../data',file_suffix='topics',num_topics=50):

    ## set number of topics. I know this is not the right way to do the things. What to do, we all carries the weights of  decisions made in the past.
    NUM_TOPICS = num_topics


    ## copy if inplace is not true
    if not inplace:
        train_df = train_df.copy()
        test_df = test_df.copy()

    ## fill na values
    train_df.fillna("",inplace=True)
    test_df.fillna("", inplace=True)


    model,dictionary = build_topic(train_df)

    topic_headers = create_topic_headers()
    ## create empty headers
    test_df['topic_headers'] = None
    train_df['topic_headers'] = None

    for index,row in train_df.iterrows():
        topics = get_topic_q1_and_q2(model,dictionary,row)
        train_df['topic_headers'].iloc[index] = topics
        #print(len(topics))
    train_df [topic_headers] = pd.DataFrame(train_df.topic_headers.values.tolist(),index=train_df.index,columns=topic_headers)
    train_df.drop(['topic_headers'],inplace=True,axis=1)
    print('Finished processing the train Topic models')
    for index, row in test_df.iterrows():
        topics = get_topic_q1_and_q2(model, dictionary, row)
        test_df['topic_headers'].iloc[index] = topics
        #print(len(topics))

    test_df[topic_headers] = pd.DataFrame(test_df.topic_headers.values.tolist(), index=test_df.index,
                                           columns=topic_headers)
    test_df.drop(['topic_headers'], inplace=True, axis=1)

    print('Finished processing the train Topic models')

    path = os.path.join(data_folder,'test_'+file_suffix)
    test_df.to_csv(path)
    path = os.path.join(data_folder,'train_'+file_suffix)
    train_df.to_csv(path)
    return train_df,test_df


def test():

    test_df = pd.read_csv('../data/test_processed.csv',nrows=2000)
    train_df = pd.read_csv('../data/train_processed.csv',nrows=2000)
    build_topics_scores(train_df,test_df)



if __name__ == '__main__':
    test()


