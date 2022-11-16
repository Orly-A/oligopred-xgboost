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

from io import StringIO

pd.set_option('display.max_columns', 20)
NUM_CV = 5


def open_scores(file_name, hyper_param_list):
    """
    Open the scores file and return a dataframe.
    split the lines to the correct columns.
    calc the average of the scores for each model
    remove models with less than 5 rows
    """
    cols = ["colsample_bylevel", "colsample_bynode", "colsample_bytree", "eta", "gamma", "max_depth",
            "min_child_weight", "n_estimators", "objective", "reg_alpha", "reg_lambda", "tree_method",
            "score_train", "score_test"]
    # use grep to get the lines that begin with [
    sys.command = "grep -e '^\[' " + file_name
    run_text = os.popen(sys.command).read()
    # print(run_text)
    # print(type(run_text))
    if run_text == '':
        return
    run_text = StringIO(run_text)

    # runlog = pd.read_table(run_text, sep=',', header=None, names=cols)
    runlog = pd.read_csv(run_text, sep=',', header=None, names=cols)

    # make sure we have the right row count
    # assert runlog.shape[1] == 8
    # print(runlog)
    runlog[['cv', 'colsample_bylevel']] = runlog['colsample_bylevel'].str.split('=', 1, expand=True)
    runlog['colsample_bylevel'] = runlog['colsample_bylevel'].astype(float)
    runlog['colsample_bynode'] = runlog['colsample_bynode'].str.split('=', 1, expand=True)[1]
    runlog['colsample_bynode'] = runlog['colsample_bynode'].astype(float)
    runlog['colsample_bytree'] = runlog['colsample_bytree'].str.split('=', 1, expand=True)[1]
    runlog['colsample_bytree'] = runlog['colsample_bytree'].astype(float)
    runlog['eta'] = runlog['eta'].str.split('=', 1, expand=True)[1]
    runlog['eta'] = runlog['eta'].astype(float)
    runlog['gamma'] = runlog['gamma'].str.split('=', 1, expand=True)[1]
    runlog['gamma'] = runlog['gamma'].astype(float)
    runlog['max_depth'] = runlog['max_depth'].str.replace(r'\D+', '', regex=True)
    runlog['max_depth'] = runlog['max_depth'].astype(int)
    # do the same for min_child_weight
    runlog['min_child_weight'] = runlog['min_child_weight'].str.replace(r'\D+', '', regex=True)
    runlog['min_child_weight'] = runlog['min_child_weight'].astype(int)
    # do the same for n_estimators
    runlog['n_estimators'] = runlog['n_estimators'].str.replace(r'\D+', '', regex=True)
    runlog['n_estimators'] = runlog['n_estimators'].astype(int)
    # for objective leave only the text on the right side of the equal sign
    runlog['objective'] = runlog['objective'].str.split('=', 1, expand=True)[1]
    runlog['reg_alpha'] = runlog['reg_alpha'].str.split('=', 1, expand=True)[1]
    runlog['reg_alpha'] = runlog['reg_alpha'].astype(float)
    runlog['reg_lambda'] = runlog['reg_lambda'].str.split('=', 1, expand=True)[1]
    runlog['reg_lambda'] = runlog['reg_lambda'].astype(float)
    # for tree_method leave only the text on the right side of the equal sign
    # then remove the ";" from the end of the string
    runlog['tree_method'] = runlog['tree_method'].str.split('=', 1, expand=True)[1]
    runlog['tree_method'] = runlog['tree_method'].str[:-1]
    # for score_train do the same as eta
    runlog['score_train'] = runlog['score_train'].str.split('=', -1, expand=True)[2]
    runlog['score_train'] = runlog['score_train'].astype(float)
    # split the score_test string into two parts, each containing a number
    runlog[['score_test', 'run_time']] = runlog['score_test'].str.split(')', 1, expand=True)
    # for score_test leave only the number on the right side of the equal sign
    runlog['score_test'] = runlog['score_test'].str.split('=', 1, expand=True)[1]
    runlog['score_test'] = runlog['score_test'].astype(float)
    # for run_time leave only the number on the right side of the equal sign
    runlog['run_time'] = runlog['run_time'].str.split('=', 1, expand=True)[1]
    runlog['run_time'] = runlog['run_time'].str.replace(r'd+\.\d+', '', regex=True)
    runlog['run_time'] = runlog['run_time'].str.split('.', 1, expand=True)[0]
    runlog['run_time'] = runlog['run_time'].astype(int)
    # for cv split on tge closed square bracket
    runlog['cv'] = runlog['cv'].str.split(']', 1, expand=True)[0]
    runlog['cv'] = runlog['cv'].str.split(' ', 1, expand=True)[1]

    runlog["mean_score_train"] = runlog.groupby(hyper_param_list)["score_train"].transform("mean")
    runlog["mean_score_test"] = runlog.groupby(hyper_param_list)["score_test"].transform("mean")
    runlog["mean_run_time"] = runlog.groupby(hyper_param_list)["run_time"].transform("mean")
    # Check how many rows for each model and remove the ones with less than 5 rows
    runlog["count_score_test"] = runlog.groupby(hyper_param_list)["score_test"].transform("count")
    runlog.drop(runlog[runlog["count_score_test"] < 5].index, inplace=True)
    # drop duplicate rows based on hyper_param_list
    runlog.drop_duplicates(hyper_param_list, inplace=True)
    return runlog


def analyze_scores_choose_params(scores_file, hyper_param_list):
    """
    open the aggregated score file
    check how many different models there are
    calculate some stats and distributions
    find the best model params
    :return: dict (?) with params to use
    """
    score_tab = scores_file.groupby(hyper_param_list)["mean_score_test"].mean().reset_index(). \
        sort_values(by=["mean_score_test"], ascending=False)
    print(score_tab.head(50))
    best_params = score_tab.iloc[0][:-1].to_dict()
    return best_params


def define_data():
    with open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/train_set.pkl", 'rb') as f:
        overall_train_set = pickle.load(f)
    # index reset is important for the stratified splitting and the saving to lists
    overall_train_set.reset_index(drop=True, inplace=True)
    overall_train_set = remove_small_groups(overall_train_set)
    X = pd.DataFrame(np.vstack(overall_train_set['embeddings']))
    y = overall_train_set["nsub"]
    groups = overall_train_set["representative"]
    cv = StratifiedGroupKFold(n_splits=NUM_CV)
    df = pd.DataFrame(np.vstack(X))
    # convert_dict = gen_converter()
    # y = y.map(convert_dict)
    return X, y, groups, cv, df


def remove_small_groups(overall_train_set):
    overall_train_set_no_embed = overall_train_set[["code", "nsub", "representative"]]
    overall_train_set2 = overall_train_set.copy()
    list_of_nsubs = list(set(overall_train_set2["nsub"].tolist()))
    for nsub in list_of_nsubs:
        num_of_clusts = overall_train_set_no_embed[overall_train_set_no_embed['nsub'] == nsub].groupby(
            "representative").nunique().shape[0]
        if num_of_clusts < NUM_CV:
            print(nsub, "nsub")
            print(num_of_clusts, "num_of_clusts")
            overall_train_set2 = overall_train_set2[overall_train_set2.nsub != nsub]
    return overall_train_set2


def train_from_hyp(best_params, X, y, groups):
    xgb_model = XGBClassifier(objective='multi:softprob')

    y = y.values.astype(int)
    le = LabelEncoder()
    y = le.fit_transform(y)

    xgb_model.set_params(**best_params)

    # generate a scorer suitable for this kind of data
    f1_score_weighted = make_scorer(f1_score, average="weighted")
    f1_score_weighted

    xgb_model.fit(X, y)

    # Save the final model
    pickle.dump(xgb_model, open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_model.pkl", "wb"))
    joblib.dump(xgb_model, "/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/xgb_random.joblib")


if __name__ == "__main__":
    # get the path to the working directory
    working_dir = "/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/log_results_fourth_tuning"
    # find all the log files in the working directory that have k5 in the name and end with .log
    log_files = [f for f in os.listdir(working_dir) if re.search(r'widerun.*.log', f) and f.endswith('.log')]
    # the next line just gets one file, for checking purposes
    # log_files = [f for f in os.listdir(working_dir) if re.search(r'k5a.log', f) and f.endswith('.log')]
    hyper_param_list = ["colsample_bylevel", "colsample_bynode", "colsample_bytree", "eta", "gamma", "max_depth",
                        "min_child_weight", "n_estimators", "objective", "reg_alpha", "reg_lambda", "tree_method"]
    # create a dataframe for the scores
    scores_file = pd.DataFrame()
    # loop over all the log files
    for log_file in log_files:
        print(log_file)
        # open the log file
        runlog = open_scores(os.path.join(working_dir, log_file), hyper_param_list)
        # append the runlog to the scores dataframe
        scores_file = scores_file.append(runlog)
    scores_file.reset_index(inplace=True)
    # save the scores dataframe to a pickle file
    scores_file.to_pickle(os.path.join(working_dir, "scores_widerun.pkl"))
    print(scores_file)
    print(scores_file.shape)
    best_params = analyze_scores_choose_params(scores_file, hyper_param_list)
    X, y, groups, cv, df = define_data()
    # train_from_hyp(best_params, X, y, groups)

