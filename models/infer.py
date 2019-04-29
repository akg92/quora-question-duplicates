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





class Infer:
    def __init__(self):
        print("do nothing")
    def get_full_path(self,path):
        return os.path.join(os.path.dirname(__file__),path)

    def compile_results(self,asked_question,questions,score):
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
    char_model = 'keras_char_tokenizer_with_MaxLen.pk'
    charembNN_model = 'Model.05-0.8240.hdf5'
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
            with open(self.char_model, 'rb') as f:
                self.char_model_loaded, self.MAX_LEN_loaded = pk.load(f)
        self.charembNN_model_loaded = self.build_model()
        self.charembNN_model_loaded.load_weights(self.charembNN_model)

    def get_char_mapping(self, questions):
        q_char_seq = self.char_model_loaded.texts_to_sequences(questions)
        return pad_sequences(q_char_seq, maxlen=self.MAX_LEN_loaded)

    def get_score(self, question, asked_question):
        return self.charembNN_model_loaded.predict([question, asked_question])

    def predict(self, question, db_questions):
        char_map_asked_question = self.get_char_mapping(question)
        char_map_db_questions = self.get_char_mapping(db_questions)

        scores = []
        for ele in char_map_db_questions:
            ele = np.reshape(ele, (1, ele.shape[0]))
            print(ele.shape)
            score = self.get_score(ele, char_map_asked_question)
            print(score)
            # print("Score{}".format(score))
            # break
            scores.append(score[0][0])
        return self.compile_results(question, db_questions, scores)






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


