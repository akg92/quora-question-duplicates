import os
import  pickle as pk
import numpy as np
import scipy
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GlobalAveragePooling1D, Lambda, \
    Bidirectional
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re




class Infer:
    def __init__(self):
        print("do nothing")
    def get_full_path(self,path):
        return os.path.join(os.path.dirname(__file__),path)

    def compile_results(self,asked_question,questions,score):
        #print('Score {}'.format(score))
        sorted_index = np.argsort(score)[::-1]
        full_result = {}
        full_result['asked'] = asked_question
        result = []
        #print(score)
        for index in sorted_index:
            temp = {}
            temp['score'] = score[index]
            temp['question'] = questions[index]
            result.append(temp)
        full_result['result'] = result
        return full_result
"""
    Infer the character encoding.
"""
class InferXGBChar(Infer):

    char_model = './trained/char_n_gram_model.h5'
    xgb_model = './trained/CharNGramXgBoost.pickle'
    xgb_model_loaded = None
    char_model_loaded = None
    def __init__(self):
        super(Infer)
        self.initilize()
    """
        Initalize the path
    """
    def initilize(self):
        self.abs_char_model = self.get_full_path(self.char_model)
        self.abs_xgb_model = self.get_full_path(self.xgb_model)

        if not self.xgb_model_loaded:
            with open(self.abs_xgb_model,'rb') as f:
                self.xgb_model_loaded = pk.load(f)

            with open(self.abs_char_model,'rb') as f:
                self.char_model_loaded = pk.load(f)

    def get_char_mapping(self,question):
        return self.char_model_loaded.transform([question])

    def get_score(self,q1,q2):
        print(q1+q2)
        return self.xgb_model_loaded.predict_proba(q1+q2)

    def get_score2(self,q):
        return self.xgb_model_loaded.predict_proba(q)

    def predict(self,question,db_questions):
        question_map = self.get_char_mapping(question)[0]
        scores = []
        for ele in db_questions:
            char_mapping = self.get_char_mapping(ele)[0]
            appended = scipy.sparse.hstack((question_map,char_mapping))
            #print(appended)
            score = self.get_score2(appended)
            #print("Score{}".format(score))
            #break
            scores.append(score[0][1])
        return self.compile_results(question,db_questions,scores)


class InferCharEmbModel(Infer):
    char_model = './trained/keras_char_tokenizer_with_MaxLen.pk'
    charembNN_model = './trained/Model.05-0.8240.hdf5'
    char_model_loaded = None
    charembNN_model_loaded = None
    MAX_LEN_loaded = None

    def __init__(self):
        super(Infer)
        self.initilize()

    def vec_distance(self, vects):
        x, y = vects
        return K.sum(K.square(x - y), axis=1, keepdims=True)

    def vec_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def build_model(self):
        nb_words = 40
        max_sentence_len = self.MAX_LEN_loaded
        embedding_layer = Embedding(nb_words, 300,
                                    input_length=max_sentence_len)
        lstm_layer = LSTM(128)
        sequence_1_input = Input(shape=(max_sentence_len,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer(embedded_sequences_1)

        sequence_2_input = Input(shape=(max_sentence_len,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        y1 = lstm_layer(embedded_sequences_2)

        distance = Lambda(self.vec_distance, output_shape=self.vec_output_shape)([x1, y1])
        dense1 = Dense(16, activation='sigmoid')(distance)
        dense1 = Dropout(0.3)(dense1)

        bn2 = BatchNormalization()(dense1)
        prediction = Dense(1, activation='sigmoid')(bn2)

        model = Model(input=[sequence_1_input, sequence_2_input], output=prediction)
        return model

    """
        Initalize the path and the network model with weights
    """

    def initilize(self):
        if not self.char_model_loaded:
            with open(self.get_full_path(self.char_model), 'rb') as f:
                self.char_model_loaded, self.MAX_LEN_loaded = pk.load(f)
        self.charembNN_model_loaded = self.build_model()
        self.charembNN_model_loaded.load_weights(self.get_full_path(self.charembNN_model))

    def get_char_mapping(self, questions):
        q_char_seq = self.char_model_loaded.texts_to_sequences(questions)
        return pad_sequences(q_char_seq, maxlen=self.MAX_LEN_loaded)

    def get_score(self, question, asked_question):
        return self.charembNN_model_loaded.predict([question, asked_question])

    def predict(self, question, db_questions):
        question = np.array([question])
        db_questions = np.array(db_questions)
        char_map_asked_question = self.get_char_mapping(question)
        char_map_db_questions = self.get_char_mapping(db_questions)

        scores = []
        for ele in char_map_db_questions:
            ele = np.reshape(ele, (1, ele.shape[0]))
            #print(ele.shape)
            score = self.get_score(ele, char_map_asked_question)
            #print(score)
            # print("Score{}".format(score))
            # break
            scores.append(score[0][0])

        scores = [x.item() for x in scores]
        return self.compile_results(question[0], db_questions, scores)




class InferConv1d(Infer):


    model =None
    model_dictionary = None
    embedding = None
    max_seq_length = 212
    model_file = './trained/weights-improvement-14-0.82.hdf5'
    embedding_file = './trained/GoogleNews-vectors-negative300.bin.gz'
    model_dictionary_file = './trained/convmodel_dictionary.txt'
    embedding_dim = 300
    """
        Constructor
    """

    def __init__(self):
        super(InferConv1d, self).__init__()
        if not self.model_dictionary:
            self.load_all()

    def get_model(self):
        #from keras_self_attention import SeqSelfAttention
        from keras.layers import Dense
        from keras import backend
        from keras.layers import TimeDistributed, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D
        from keras.callbacks import ModelCheckpoint
        # Model variables
        backend.clear_session()
        n_hidden = 50
        gradient_clipping_norm = 1.25
        batch_size = 128
        n_epoch = 100

        def exponent_neg_manhattan_distance(left, right):
            ''' Helper function for the similarity estimate of the LSTMs outputs'''
            return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

        # The visible layer
        left_input = Input(shape=(self.max_seq_length,), dtype='int32')
        right_input = Input(shape=(self.max_seq_length,), dtype='int32')

        embedding_layer = Embedding(86002, self.embedding_dim, input_length=self.max_seq_length)

        # Embedded version of the inputs
        #encoded_left = embedding_layer(left_input)
        #encoded_right = embedding_layer(right_input)

        # Since this is a siamese network, both sides share the same LSTM
        # shared_lstm = LSTM(n_hidden,return_sequences=True)

        # left_output = shared_lstm(encoded_left)
        # right_output = shared_lstm(encoded_right)

        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        ## conv12
        conv = Conv1D(filters=1500, kernel_size=4, padding='valid', activation='sigmoid', strides=1)

        encoded_left = conv(encoded_left)
        encoded_right = conv(encoded_right)

        pooling = MaxPooling1D(pool_size=4)
        encoded_left = pooling(encoded_left)
        encoded_right = pooling(encoded_right)

        conv2 = Conv1D(filters=3000, kernel_size=4, padding='valid', activation='sigmoid', strides=1)
        encoded_left = conv2(encoded_left)
        encoded_right = conv2(encoded_right)

        pooling2 = GlobalMaxPooling1D()
        encoded_left = pooling2(encoded_left)
        encoded_right = pooling2(encoded_right)

        dense = Dense(256)
        left_output = dense(encoded_left)
        right_output = dense(encoded_right)

        # Calculates the distance as defined by the MaLSTM model
        malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                                 output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        # Pack it all up into a model
        malstm = Model([left_input, right_input], [malstm_distance])
        return malstm



    def load_all(self):


        ## vocabulary file
        with open(self.get_full_path(self.model_dictionary_file),'rb') as f:
            self.model_dictionary = pk.load(f)
        ## model load
        self.model = self.get_model()
        self.model.load_weights(self.get_full_path(self.model_file))
        self.model._make_predict_function()

        ## Embedding load

    def text_to_word_list(self,text):
        ''' Pre process and convert texts to a list of words '''
        text = str(text)
        text = text.lower()

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        text = text.split()
        return text


    def w2vec(self,text):
        words = self.text_to_word_list(text)
        inverted_mapping = np.zeros(self.max_seq_length,dtype=np.int32)

        i = 211
        for word in words:
            if word in self.model_dictionary:
                inverted_mapping[i] = self.model_dictionary[word]
            i-=1
            if i==-1:
                break
        return inverted_mapping

    def predict(self,asked_q,questions):
        asked_mapping =self.w2vec(asked_q)
        questions_mappings = []
        for question in questions:
            questions_mappings.append(self.w2vec(question))
            #print(self.w2vec(question))

        ## get np mapping
        input_q1 = np.tile(asked_mapping,(len(questions),1))
        input_q2 = np.array(questions_mappings,dtype=np.int32)

        print('Shapes {}:{}'.format(input_q1.shape,input_q2.shape))

        scores = self.model.predict([input_q1,input_q2])
        scores = [x.item() for x in np.ravel(scores)]
        return self.compile_results(asked_q, questions, scores)




if __name__ == '__main__':
    model = InferXGBChar()
    questions = ["How is information retrieval course in tamu?","Information retrieval content","How are you today"]
    asked_q = "Can I take Information retrieval course?"
    result = model.predict(asked_q,questions)
    print(result)

    # TO Infer from Char Emb Model
    model = InferCharEmbModel()
    questions = ["How is information retrieval course in tamu?", "Information retrieval content", "How are you today"]
    asked_q = "Can I take Information retrieval course?"
    result = model.predict(np.array([asked_q]), np.array(questions))
    print(result)


