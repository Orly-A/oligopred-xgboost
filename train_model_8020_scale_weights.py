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

# HEAD_OF_DATA = 1000
# LR = 0.1
NUM_CV = 5
# NUM_ITER = 20




def data_definition(overall_train_set):
    print(sorted(overall_train_set.nsub.unique().tolist()))
    overall_train_set = remove_small_groups(overall_train_set)
    print("removed small groups")
    print(sorted(overall_train_set.nsub.unique().tolist()))

    overall_train_set = calc_weights(overall_train_set)
    # print(overall_train_set)

    overall_train_set.reset_index(drop=True, inplace=True)
    # print(overall_train_set)
    print(sorted(overall_train_set.nsub.unique().tolist()))
    X = overall_train_set["code"]
    y = overall_train_set["nsub"]
    groups = overall_train_set["representative"]

    cv = StratifiedGroupKFold(n_splits=NUM_CV, shuffle=True, random_state=1)
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
    weights = train_set['weights']

    X_test = test_set['embeddings'].tolist()
    y_test = test_set['nsub']

    pickle.dump(X_train, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/weights_X_train_80.pkl", "wb"))
    pickle.dump(y_train, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/weights_y_train_80.pkl", "wb"))
    pickle.dump(X_test, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/weights_X_test_20.pkl", "wb"))
    pickle.dump(y_test, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/weights_y_test_20.pkl", "wb"))
    # print(y_train)
    # print(y_test)


    return X_train, y_train, X_test, y_test, weights


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


def calc_weights(overall_train_set):
    nsub_dict = {}
    nsub_list = sorted(overall_train_set.nsub.unique().tolist())
    for nsub in nsub_list:
        # print(nsub, overall_train_set[overall_train_set["nsub"] == nsub].shape[0])
        nsub_dict[nsub]=overall_train_set[overall_train_set["nsub"] == nsub].shape[0]
    for k, v in nsub_dict.items():
        nsub_dict[k] = round(20/v, 4)
    overall_train_set["weights"] = np.sqrt(overall_train_set.nsub.map(nsub_dict))
#    overall_train_set["weights"] = overall_train_set.nsub.map(nsub_dict)

    print(overall_train_set["weights"])
    return overall_train_set


def initial_xgboost_classifier(X_train, y_train, X_test, y_test, weights):
    xgb_class = xgb.XGBClassifier(objective='multi:softprob', eta=0.4, max_depth=6, min_child_weight=9,
                                  n_estimators=1500, tree_method="approx", random_state=5)

    print(y_train.unique())
    print(y_test.unique())

    y_train = y_train.values.astype(int)
    y_test = y_test.values.astype(int)
    le = LabelEncoder()
    y_test = le.fit_transform(y_test)
    y_train = le.fit_transform(y_train)
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_dict)

    with open('le_dict.pkl', 'wb') as f:
        pickle.dump(le_dict, f)

    print("starting the tuning")
    print(datetime.datetime.now())
    xgb_class.fit(X_train, y_train, sample_weight=weights)
    preds = xgb_class.predict(X_test)

    pickle.dump(xgb_class, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_8020_weights_sqrt_random5.pkl", "wb"))
    joblib.dump(xgb_class, "/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_8020_weights_sqrt_random5.joblib")
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

    X_train, y_train, X_test, y_test, weights = data_definition(overall_train_set)
    initial_xgboost_classifier(X_train, y_train, X_test, y_test, weights)
