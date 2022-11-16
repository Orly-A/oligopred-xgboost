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
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler


import xgboost as xgb
from xgboost import XGBClassifier

# HEAD_OF_DATA = 1000
# LR = 0.1
NUM_CV = 5
# NUM_ITER = 20
UNDERSAMPLE_FACTOR = 3




def data_definition(overall_train_set):
    overall_train_set = remove_small_groups(overall_train_set)
    overall_train_set.reset_index(drop=True, inplace=True)
    overall_train_set = downsample_mjorities(overall_train_set)
    overall_train_set.reset_index(drop=True, inplace=True)
    X = overall_train_set["code"]
    y = overall_train_set["nsub"]
    groups = overall_train_set["representative"]

    cv = StratifiedGroupKFold(n_splits=NUM_CV)
    train_lst = []
    test_lst = []
    for train_idxs, test_idxs in cv.split(X, y, groups):
        train_lst.append(X[train_idxs].tolist())
        test_lst.append(X[test_idxs].tolist())

    train_idx_df = pd.DataFrame(train_lst).transpose()
    train_idx_df.rename(columns={0:"train_0", 1:"train_1", 2:"train_2", 3:"train_3", 4:"train_4"}, inplace=True)
    test_idx_df = pd.DataFrame(test_lst).transpose()
    test_idx_df.rename(columns={0:"test_0", 1:"test_1", 2:"test_2", 3:"test_3", 4:"test_4"}, inplace=True)

    merged_train_test = pd.concat([train_idx_df, test_idx_df], axis=1, join="outer")
    train_set = overall_train_set[overall_train_set["code"].isin(merged_train_test["train_0"])]
    test_set = overall_train_set[overall_train_set["code"].isin(merged_train_test["test_0"])]
    X_train = train_set['embeddings'].tolist()
    y_train = train_set['nsub']

    X_test = test_set['embeddings'].tolist()
    y_test = test_set['nsub']

    pickle.dump(X_train, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/X_train_80_downsample.pkl", "wb"))
    pickle.dump(y_train, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/y_train_80_downsample.pkl", "wb"))
    pickle.dump(X_test, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/X_test_20_downsample.pkl", "wb"))
    pickle.dump(y_test, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/y_test_20_downsample.pkl", "wb"))


    return X_train, y_train, X_test, y_test


def remove_small_groups(overall_train_set):
    overall_train_set_no_embed = overall_train_set[["code", "nsub", "representative"]]
    overall_train_set2 = overall_train_set.copy()
    list_of_nsubs = list(set(overall_train_set2["nsub"].tolist()))
    for nsub in list_of_nsubs:
        num_of_clusts = overall_train_set_no_embed[overall_train_set_no_embed['nsub'] == nsub].groupby("representative").nunique().shape[0]
        if num_of_clusts < NUM_CV:
            print(nsub, "nsub")
            print(num_of_clusts, "num_of_clusts")
            overall_train_set2 = overall_train_set2[overall_train_set2.nsub != nsub]
    return overall_train_set2


def downsample_mjorities(overall_train_set):
    new_1_count = int(overall_train_set[overall_train_set["nsub"] == 1].shape[0]/UNDERSAMPLE_FACTOR)
    new_2_count = int(overall_train_set[overall_train_set["nsub"] == 2].shape[0]/UNDERSAMPLE_FACTOR)
    under_sample_dict = {1: new_1_count, 2: new_2_count}
    list_of_nsubs = list(set(overall_train_set["nsub"].tolist()))
    list_of_nsubs.remove(1)
    list_of_nsubs.remove(2)
    for nsub in list_of_nsubs:
        counter = int(overall_train_set[overall_train_set["nsub"] == nsub].shape[0])
        under_sample_dict[nsub] = counter
    print(under_sample_dict)
    rus = RandomUnderSampler(random_state=1, sampling_strategy=under_sample_dict)
    X, y = rus.fit_resample(overall_train_set[["code"]], overall_train_set["nsub"])
    overall_train_set = overall_train_set[overall_train_set.code.isin(X["code"].tolist())]
    return overall_train_set


def initial_xgboost_classifier(X_train,y_train, X_test=None, y_test=None):
    xgb_class = xgb.XGBClassifier(objective='multi:softprob', eta=0.2, max_depth=6, min_child_weight=9,
                                  n_estimators=1500, tree_method="auto", random_state=1)

    y_train = y_train.values.astype(int)
    y_test = y_test.values.astype(int)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    print("starting the tuning")
    print(datetime.datetime.now())
    xgb_class.fit(X_train, y_train)
    preds = xgb_class.predict(X_test)

    pickle.dump(xgb_class, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_8020_downsample.pkl", "wb"))
    joblib.dump(xgb_class, "/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_8020_downsample.joblib")
    print("finished the tuning")
    print(datetime.datetime.now())

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % rmse)
    print('Precision: %.3f' % precision_score(y_test, preds, average='weighted'))
    print('Recall: %.3f' % recall_score(y_test, preds, average='weighted'))
    print('F-measure: %.3f' % f1_score(y_test, preds, average='weighted'))
    print("adjusted Balanced accuracy: %.3f" % metrics.balanced_accuracy_score(y_test, preds, adjusted=True))


if __name__ == "__main__":
    # load the tab used for embedding, only the training set of course
    with open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/train_set.pkl", 'rb') as f:
        overall_train_set = pickle.load(f)
    # index reset is important for the stratified splitting and the saving to lists
    overall_train_set.reset_index(drop=True, inplace=True)
    X_train, y_train, X_test, y_test = data_definition(overall_train_set)
    initial_xgboost_classifier(X_train, y_train, X_test, y_test)
