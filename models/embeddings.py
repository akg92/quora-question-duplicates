
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
    Process the text to embedding.
    """

    def get_embedding(self,text,single_word=False):


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

        return result


