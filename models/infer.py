import os
import  pickle as pk
import numpy as np
import scipy
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






if __name__ == '__main__':
    model = InferXGBChar()
    questions = ["How is information retrieval course in tamu?","Information retrieval content","How are you today"]
    asked_q = "Can I take Information retrieval course?"
    result = model.predict(asked_q,questions)
    print(result)

