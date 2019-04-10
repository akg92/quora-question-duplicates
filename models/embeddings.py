
import numpy as np
from gensim.models.doc2vec import Doc2Vec

class Embedding_Helper:

    model = None
    dimension = -1
    def __init__(self,model_type='glove'):
        if model_type=="glove" and Embedding_Helper.model is not None:
            Embedding_Helper.model = self.load_glove()
    """
    Wordembedding.
    Change the file for changing the location
    """

    def load_glove(self,glove_file='../data/glove.6B.50d'):

        glove_model = {}
        with open(glove_file) as f:
            for line in f:
                words = line.split(" ")
                glove_model[words[0]] = [float(x) for x in words[1:]]  ## the mapping is the word, dimension
        Embedding_Helper.dimension = 50
        return glove_model

    """
    Process the text to embedding. set avg=False for if you want encoding for each word
    """

    def get_embedding(self,text,single_word=False,avg=True):


        if single_word:
            if text in Embedding_Helper.model:
                return Embedding_Helper.model[text]
        else:
            result = []
            ## find for each word
            for word in text.split():
                if word in Embedding_Helper.model:
                    result.append(Embedding_Helper.model[word])
                else:
                    result.append([0 for x in range(Embedding_Helper.dimension)])

        if avg:
            temp = np.array(result)
            result = np.mean(temp,axis=0)
        return result



    def __del__(self):
        if self.model:
            del self.model

"""
Currently using the pre-trained model only. 
Depends on the performance we have to train the model ourselves.
"""
class DocEmbedding():

    model = None
    dimnesions = 50
    def __init__(self,model_file='../doc2vec.model.txt'):
        ## Reuse the loaded model
        if not DocEmbedding.model:
            return
        self.model = Doc2Vec.load(model_file)
    """
        Create header 
    """
    def create_header(self):
        return ['emb_q1_'+str(x) for x in range(self.dimension)] + ['emb_q2_'+str(x) for x in range(self.dimension)]


    def doc2vec(self,df,inplace=True):
        ## create copy if not inplace
        if not inplace:
            df = df.copy()

        headers =  self.create_header()
        ## default value is set to zero. cheange if required
        df[headers] = 0

        for index,row in df.iterrows():
            q1_emb = self.model.infer_vector(row.question1.split())
            q2_emb = self.model.infer_vector(row.question2.split())
            df[headers].iloc[index] = q1_emb+q2_emb

        return df









