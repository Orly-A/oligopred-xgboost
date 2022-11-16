import re
import os
import pickle
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import datetime
import joblib

from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.dummy import DummyClassifier

import xgboost as xgb
from xgboost import XGBClassifier

NUM_CV = 5

def open_data(arg_pars):
    path_2_X_train = arg_pars[0]
    path_2_y_train = arg_pars[1]
    path_2_X_test = arg_pars[2]
    path_2_y_test = arg_pars[3]
    X_train = pickle.load(open(path_2_X_train, "rb"))
    y_train = pickle.load(open(path_2_y_train, "rb"))
    X_test = pickle.load(open(path_2_X_test, "rb"))
    y_test = pickle.load(open(path_2_y_test, "rb"))
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, X_test, y_test):

    y_train = y_train.values.astype(int)
    y_test = y_test.values.astype(int)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)


    clf_dum = DummyClassifier(strategy='stratified', random_state=1)
    clf_dum.fit(X_train, y_train)

    preds = clf_dum.predict(X_test)
    pickle.dump(clf_dum, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/clf_dum.pkl", "wb"))
    joblib.dump(clf_dum, "/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/clf_dum.joblib")
    print(datetime.datetime.now())

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % rmse)
    print('Precision: %.3f' % precision_score(y_test, preds, average='weighted'))
    print('Recall: %.3f' % recall_score(y_test, preds, average='weighted'))
    print('F-measure: %.3f' % f1_score(y_test, preds, average='weighted'))
    print("adjusted Balanced accuracy: %.3f" % metrics.balanced_accuracy_score(y_test, preds, adjusted=True))


    conf_mat = metrics.confusion_matrix(y_test, preds)
    conf_mat_df = pd.DataFrame(conf_mat)

    with open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/le_dict.pkl", 'rb') as f:
        le_dict = pickle.load(f)
    inv_map = {v: k for k, v in le_dict.items()}
    conf_mat_df.rename(inv_map, axis=1, inplace=True)
    conf_mat_df.rename(inv_map, inplace=True)
    pickle.dump(conf_mat_df, open("clf_dum_conf_mat_df.pkl", "wb"))
    conf_mat_df_percent = conf_mat_df.div(conf_mat_df.sum(axis=1), axis=0).round(2)

if __name__ == "__main__":
    arg_pars = sys.argv[1:]
    X_train, y_train, X_test, y_test = open_data(arg_pars)
    train_model(X_train, y_train, X_test, y_test)
