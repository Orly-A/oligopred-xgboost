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
from imblearn.under_sampling import RandomUnderSampler


import xgboost as xgb
from xgboost import XGBClassifier

HEAD_OF_DATA = 1000
LR = 0.1
NUM_CV = 5
NUM_ITER = 100
UNDERSAMPLE_FACTOR = 3


# load the tab used for embedding, only the training set of course
with open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/train_set.pkl", 'rb') as f:
    overall_train_set = pickle.load(f)
# index reset is important for the stratified splitting and the saving to lists
overall_train_set.reset_index(drop=True, inplace=True)

def data_definition_hyp():
    overall_train_set = remove_small_groups()
    overall_train_set = downsample_mjorities(overall_train_set)
    X = pd.DataFrame(np.vstack(overall_train_set['embeddings']))
    y = overall_train_set["nsub"]
    groups = overall_train_set["representative"]
    cv = StratifiedGroupKFold(n_splits=NUM_CV)
    df = pd.DataFrame(np.vstack(X))
    # convert_dict = gen_converter()
    # y = y.map(convert_dict)
    return X, y, groups, cv, df


def remove_small_groups():
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



def use_head_of_data(overall_train_set):
    # for understanding how to work with the model
    overall_train_set = overall_train_set[:HEAD_OF_DATA]
    # the next two lines are for avoiding errors due to small groups of data, removed manually...
    overall_train_set= overall_train_set[overall_train_set.nsub != 7.0]
    overall_train_set= overall_train_set[overall_train_set.nsub != 14.0]

    X = pd.DataFrame(np.vstack(overall_train_set['embeddings']))
    y = overall_train_set["nsub"]
    groups = overall_train_set["representative"]
    cv = StratifiedGroupKFold(n_splits=NUM_CV)
    df = pd.DataFrame(np.vstack(X))
    # convert_dict = gen_converter()
    # y = y.map(convert_dict)

    return X, y, groups, cv, df


def initial_xgboost_classifier(X_train,y_train, X_test=None, y_test=None):
# flow (and some params) from here: https://www.datacamp.com/community/tutorials/xgboost-in-python
    xgb_class = xgb.XGBClassifier(objective='multi:softprob', learning_rate=LR,
                    max_depth=5, n_estimators=100, random_state=1)
    y_train = y_train.values.astype(int)
    le = LabelEncoder()
    y_train = le.fit_transform(y)
    print("starting the tuning")
    print(datetime.datetime.now())


    xgb_class.fit(X_train, y_train)
    # preds = xgb_class.predict(X_test)
    # rmse = np.sqrt(mean_squared_error(y_test, preds))
    # print("RMSE: %f" % rmse)
    # print('Precision: %.3f' % precision_score(y_test, preds, average='weighted'))
    # print('Recall: %.3f' % recall_score(y_test, preds, average='weighted'))
    # print('F-measure: %.3f' % f1_score(y_test, preds, average='weighted'))
    # print("adjusted Balanced accuracy: %.3f" % metrics.balanced_accuracy_score(y_test, preds, adjusted=True))
    print("finished the tuning")
    print(datetime.datetime.now())

    pickle.dump(xgb_class, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_model.pkl", "wb"))
    joblib.dump(xgb_class, "/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_random.joblib")

def generate_grid():
    # Code from here:
    # https://towardsdatascience.com/cross-validation-and-hyperparameter-tuning-how-to-optimise-your-machine-learning-model-13f005af9d7d
    # one can set the feature_weights for DMatrix to define
    # the probability of each feature being selected when using column sampling.
    # There’s a similar parameter for fit method in sklearn interface.

    colsample_bytree = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    colsample_bylevel = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    colsample_bynode = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Number of trees to be used
    xgb_n_estimators = [int(x) for x in np.arange(50, 850, 200)]
    xgb_n_estimators.extend([1000, 1500, 2000])
    # Maximum number of levels in tree
    xgb_max_depth = [int(x) for x in np.linspace(2, 20, 10)]
    xgb_max_depth = [round(x, 1) for x in xgb_max_depth]
    # Minimum number of instances needed in each node
    xgb_min_child_weight = [int(x) for x in np.linspace(1, 10, 10)]
    # Tree construction algorithm used in XGBoost
    xgb_tree_method = ['auto', 'exact', 'approx', 'hist']
    # Learning rate
    xgb_eta = [x for x in np.linspace(0.1, 0.6, 6)]
    xgb_eta = [round(x, 1) for x in xgb_eta]
    # Minimum loss reduction required to make further partition
    reg_lambda = [0, 0.01, 0.1, 0.5, 1, 5, 10, 50, 75, 100]
    # Learning objective used
    xgb_objective = ['multi:softprob']
    # Create the grid
    xgb_grid = {"colsample_bytree": colsample_bytree,
                "colsample_bylevel": colsample_bylevel,
                "colsample_bynode": colsample_bynode,
                'n_estimators': xgb_n_estimators,
                'max_depth': xgb_max_depth,
                'min_child_weight': xgb_min_child_weight,
                'tree_method': xgb_tree_method,
                'eta': xgb_eta,
                'objective': xgb_objective,
                'reg_lambda': reg_lambda}

    return xgb_grid


def hyperparam_search(xgb_grid, groups, X, y):
    # Code from here:
    # https://towardsdatascience.com/cross-validation-and-hyperparameter-tuning-how-to-optimise-your-machine-learning-model-13f005af9d7d
    y = y.values.astype(int)
    le = LabelEncoder()
    y = le.fit_transform(y)

    xgb_base = XGBClassifier(objective='multi:softprob')#, learning_rate=LR

    print("starting the tuning")
    print(datetime.datetime.now())
    cv = StratifiedGroupKFold(n_splits=NUM_CV)
    # generate a scorer suitable for this kind of data
    f1_score_weighted = make_scorer(f1_score, average="weighted")
    f1_score_weighted
    adj_balanced_acc = make_scorer(metrics.balanced_accuracy_score, adjusted=True)
    adj_balanced_acc
    # Create the random search
    xgb_random = RandomizedSearchCV(estimator=xgb_base, param_distributions=xgb_grid,
                                    n_iter=NUM_ITER, cv=cv, verbose=4,
                                    n_jobs=-1,
                                    scoring={"f1_score": f1_score_weighted, "adjusted_balanced_accuracy": adj_balanced_acc},
                                    return_train_score=True,
                                    refit="f1_score") #, pre_dispatch=NUM_ITER, random_state=1
    # Fit the random search model
    # print("starting the fit")
    xgb_random.fit(X, y, groups=groups)
    # Get the optimal parameters
    xgb_random.best_params_

    # joblib.dump(xgb_random, "/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_random.joblib")
    # xgb_random = joblib.load("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_random.joblib")
    print("finished the tuning")
    print(xgb_random.best_score_)
    print(xgb_random.scorer_)
    print(datetime.datetime.now())
    print(xgb_random.best_params_)
    print(NUM_ITER, "number of iterations")
    print(NUM_CV, "number of k-fold")

    return xgb_random.best_params_

def train_from_hyp(best_params, X, y, groups):
    xgb_model = XGBClassifier(objective='multi:softprob')

    y = y.values.astype(int)
    le = LabelEncoder()
    y = le.fit_transform(y)

    xgb_model.set_params(**best_params)

    # generate a scorer suitable for this kind of data
    f1_score_weighted = make_scorer(f1_score, average="weighted")
    f1_score_weighted

    xgb_model.fit(X, y, groups=groups)

    #Save the final model
    pickle.dump(xgb_model, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_model.pkl", "wb"))
    joblib.dump(xgb_model, "/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_random.joblib")


def k_fold(X, y, groups):
    # k-fold cross val without tuning
    # fit xgboost on an imbalanced classification dataset
    y = y.values.astype(int)
    le = LabelEncoder()
    y = le.fit_transform(y)

    cv = StratifiedGroupKFold(n_splits=NUM_CV, random_state=1, shuffle=True)
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    xgb_model = xgb.XGBClassifier(objective='multi:softprob', cv=cv, groups=groups, random_state=1)
    # xgb_model.fit(X, y)
    # y_pred = xgb_model.predict(X)
    f1_score_weighted = make_scorer(f1_score, average="weighted")
    f1_score_weighted
    scores = cross_val_score(xgb_model, X, y, cv=cv, n_jobs=-1, scoring=f1_score_weighted, error_score='raise', groups=groups)
    # summarize performance
    # print(y_true)
    # print(y_score)
    print('f1_score_weighted: %.5f' % np.mean(scores))
    # pickle.dump(xgb_model, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_model.pkl", "wb"))
    # joblib.dump(xgb_model, "/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_random.joblib")


if __name__ == "__main__":
    # X, y, groups, cv, df = use_head_of_data(overall_train_set)
    X, y, groups, cv, df = data_definition_hyp()
    xgb_grid = generate_grid()
    best_params = hyperparam_search(xgb_grid, groups, X, y)
    print(best_params)
    # train_from_hyp(best_params, X, y, groups)
    #
    # X, y, groups, cv, X_train, y_train, y_test, df = data_definition(overall_train_set)
    # initial_xgboost_classifier(X_train, y_train, y_test)

    # X, y, groups, cv, X_train, y_train, X_test, y_test, df = data_definition()
    # initial_xgboost_classifier(X_train,y_train, X_test, y_test)
    # k_fold(X, y, groups)


