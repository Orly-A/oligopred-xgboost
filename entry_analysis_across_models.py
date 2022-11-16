import re
import os
import pickle
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from datetime import date
import joblib
# import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder


today = str(date.today())

"""
This script analyzes the results of the xgboost models.
Specifically we want to examine the variations of the performance of the models,
and check the consistency and change **per-entry**
"""

def get_models_from_main_dir():
    """
    This function returns a list of all the models in the relevant dir.
    """
    models = []
    for root, dirs, files in os.walk("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/trained_models/"):
        for file in files:
            # if file has 8020 in its name and ends with joblib, then it is a model
            if "8020" in file and file.endswith(".joblib"):
            # if file.endswith(".joblib"):
                models.append(os.path.join(root, file))
    return models


def open_train_test_sets(path):
    """
    This function opens the data from the path.
    """
    X_train = pd.read_pickle(path + "X_train_80.pkl")
    y_train = pd.read_pickle(path + "y_train_80.pkl")
    X_test = pd.read_pickle(path + "X_test_20.pkl")
    y_test = pd.read_pickle(path + "y_test_20.pkl")
    return X_train, y_train, X_test, y_test


def get_prediction_from_model(model, X_test):
    """
    This function returns the prediction of the model on the test set.
    """
    model = joblib.load(model)
    y_pred = model.predict(X_test)
    return y_pred


def get_label_mapping_dict(y_pred):
    with open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/le_dict.pkl", 'rb') as f:
        le_dict = pickle.load(f)
    inv_map = {v: k for k, v in le_dict.items()}
    y_pred_reverted = []
    for i in y_pred:
        y_pred_reverted.append(inv_map[i])
    return y_pred_reverted


def generate_results_df(y_test_for_decode, models):
    results_df = pd.DataFrame()
    results_df["nsub"] = y_test_for_decode
    for model in models:
        model_name = model.split("/")[-1].split(".")[0]
        y_pred = get_prediction_from_model(model, X_test)
        y_pred_reverted = get_label_mapping_dict(y_pred)
        results_df[model_name] = y_pred_reverted
    results_df["success"] = results_df.apply(classify_successes, axis=1)
    return results_df


def classify_successes(row):
    if (row.nsub == row.xgb_8020_weights) and (row.nsub == row.xgb_joblib8020_max_delta_1) and (row.nsub == row.xgb_8020_weights_sqrt) and (row.nsub == row.xgb_8020_800_est) and (row.nsub == row.xgb_joblib8020_max_delta_5):
        return 1
    elif (row.nsub != row.xgb_8020_weights) and (row.nsub != row.xgb_joblib8020_max_delta_1) and (row.nsub != row.xgb_8020_weights_sqrt) and (row.nsub != row.xgb_8020_800_est) and (row.nsub != row.xgb_joblib8020_max_delta_5):
        return 3
    else:
        return 2


#def analyze_


if __name__ == "__main__":
    #first arg is the path to the sets
    arg_pars = sys.argv[1:]
    le = LabelEncoder()
    X_train, y_train, X_test, y_test = open_train_test_sets(arg_pars[0])
    # y_train = y_train.values.astype(int)
    # y_train = le.fit_transform(y_train)
    y_test_for_decode = y_test.values.astype(int)
    y_test_transformed = le.fit_transform(y_test_for_decode)
    models_list = get_models_from_main_dir()
    results_df = generate_results_df(y_test_for_decode, models_list)
    results_df.to_csv("model_comparison_results_df_06082022.csv")

    print(results_df)

    print("end")




# model_comparison_results_df = pd.read_csv("model_comparison_results_df_06082022.csv")
# t = model_comparison_results_df.groupby(["success", "nsub"])["xgb_8020_weights"].count()
# pd.pivot(t, index="success", columns="nsub").plot(kind="bar", stacked=True)

