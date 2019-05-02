import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


def get_all_similarities(data_dir='../data',file_suffix='idf_appended'):
    test_fp = os.path.join(data_dir,'test_'+file_suffix)
    train_fp = os.path.join(data_dir,'train_'+file_suffix)
    train_df = pd.read_csv(train_fp)
    test_df = pd.read_csv(test_fp)
    return train_df,test_df


def extract_input_and_labels_df(append_df):
    # Drop identifier columns
    id_columns = ['id','qid1','qid2','question1','question2']
    dropped_df = append_df.drop(columns=id_columns)
    # Get features
    y_labels = append_df['is_duplicate'].values
    dropped_df.drop(columns=['is_duplicate'],inplace=True)
    X_features = dropped_df.values
    return X_features,y_labels


def svc_param_selection(X, y, nfolds):
    #Cs = [0.001, 0.01, 0.1, 1, 10]
    Cs = [0.1]
    #gammas = [0.001, 0.01, 0.1, 1]
    gammas = [0.01]
    #kernels = ['rbf','linear']
    kernels = ['rbf']
    param_grid = {'C': Cs, 'gamma' : gammas,'kernel':kernels}
    print('Starting Grid Search:')
    grid_search = GridSearchCV(SVC(), param_grid, n_jobs=-1,verbose=2)
    grid_search.fit(X, y)
    return grid_search.best_params_,grid_search.best_estimator_


train_df,test_df = get_all_similarities()
X_train,Y_train = extract_input_and_labels_df(train_df)
X_test,Y_test = extract_input_and_labels_df(test_df)
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)


#grid_search_params,best_svm_model = svc_param_selection(X_train,Y_train,5)

#dump(best_svm_model,'SVM_TF_IDF_MODEL.joblib')


#clf = load('SVM_TF_IDF_MODEL.joblib')

#y_pred = clf.predict(X_test)
#print(accuracy_score(Y_test,y_pred))








