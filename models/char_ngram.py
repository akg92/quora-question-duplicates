from models import utils
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
import scipy
import xgboost as xgb
import pickle


def extract_input_and_labels_df(df):
    # Drop identifier columns
    id_columns = ['id','qid1','qid2','question1','question2']
    dropped_df = append_df.drop(columns=id_columns)
    # Get features
    y_labels = append_df['is_duplicate'].values
    dropped_df.drop(columns=['is_duplicate'],inplace=True)
    X_features = dropped_df.values
    return X_features,y_labels


train_df,test_df = utils.get_processed_df()
uniq_questions = pd.concat((train_df['question1'],train_df['question2'])).unique()
#uniq_questions = [x for x in uniq_questions if str(x) != 'nan']
#print('Number of Unique Questions:')
#print(len(uniq_questions))
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

print('Forming TF-IDF-Char-Ngram-Features:')
tfidf_vect_ngram_chars.fit(uniq_questions)
feature_names = tfidf_vect_ngram_chars.get_feature_names()
print(feature_names)


print('Running Train and Test Transforms:')
trainq1 = train_df['question1'].values
#trainq1 = [x for x in trainq1 if str(x) != 'nan']
#print(len(trainq1))

trainq2 = train_df['question2'].values

#trainq2 = [x for x in trainq2 if str(x) != 'nan']
#print(len(trainq2))

testq1 = test_df['question1'].values
#testq1 = [x for x in testq1 if str(x) != 'nan']


testq2 = test_df['question2'].values
#testq2 = [x for x in testq2 if str(x) != 'nan']

trainq1_trans = tfidf_vect_ngram_chars.transform(trainq1)
trainq2_trans = tfidf_vect_ngram_chars.transform(trainq2)
print('Running Test Transform:')
testq1_trans = tfidf_vect_ngram_chars.transform(testq1)
testq2_trans = tfidf_vect_ngram_chars.transform(testq2)

#diff_n_rows = trainq2_trans.shape[0] - trainq1_trans.shape[0]

#trainq1_trans = scipy.sparse.vstack((trainq1_trans, scipy.sparse.csr_matrix((diff_n_rows, trainq1_trans.shape[1]))))

#diff_n_rows = testq2_trans.shape[0] - testq1_trans.shape[0]
#testq1_trans = scipy.sparse.vstack((testq1_trans, scipy.sparse.csr_matrix((diff_n_rows, testq1_trans.shape[1]))))


X_train = scipy.sparse.hstack((trainq1_trans,trainq2_trans))
X_test = scipy.sparse.hstack((testq1_trans,testq2_trans))


y_train = train_df['is_duplicate'].values
print('train shape:')
print(X_train.shape)
print(y_train.shape)
print('Test Shape:')
print(X_test.shape)
y_test = test_df['is_duplicate'].values
print(y_test.shape)
print('Training XGB Classifier from new features:')
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train)
# Saving XGB model
print('Saving Model:')
#xgb_model.save_model('CharNgramXgBoost.model')
pickle.dump(xgb_model,open("CharNGramXgBoost.pickle","wb"))

# Load saved model
#bst = xgb.Booster({'nthread':4}) #init model
print('Loading it back:')
xgb_model_loaded = pickle.load(open("CharNGramXgBoost.pickle","rb"))
print('Predicting on test set:')
xgb_prediction = xgb_model_loaded.predict(X_test)
print('character level tf-idf training score:', f1_score(y_train, xgb_model_loaded.predict(X_train), average='macro'))
print('character level tf-idf validation score:', f1_score(y_test, xgb_model_loaded.predict(X_test), average='macro'))
print(classification_report(y_test, xgb_prediction))



