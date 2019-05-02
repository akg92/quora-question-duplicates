from joblib import dump, load
import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC


clf = load('SVM_TF_IDF_MODEL.joblib')
print(type(clf))