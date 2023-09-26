import re
import os
import pickle
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from datetime import date
import joblib

import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder



"""
This script is used to analyze the results of the xgboost model.
the first arg is the wanted model full path
then 2,3,4,5 are X_train, y_train, X_test, y_test
"""


arg_pars = sys.argv[1:]
# get the model name from the args
model_name = arg_pars[0].split("/")[-1].split(".")[0]
today = str(date.today())
dest_path = "/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/analyzing_models/analyze_model_" + model_name + "_" + today + "/"

# check if the directory exists, if not create it
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

with open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/train_set.pkl", 'rb') as f:
    overall_train_set = pickle.load(f)
# index reset is important for the stratified splitting and the saving to lists
overall_train_set.reset_index(drop=True, inplace=True)

overall_train_set_no_embed = overall_train_set.drop(["embeddings"], axis=1)
a = overall_train_set_no_embed.groupby("representative").nunique("nsub").groupby("nsub").size().plot\
    (kind='bar', grid=False, log=True, color="maroon", fontsize=10,
     title="different oligomeric states per sequence similarity cluster", xlabel="different oligomeric states", ylabel="number of clusters")
a.figure.savefig(dest_path + "different_oligomeric_states_per_cluster.png")
b = overall_train_set_no_embed.groupby("representative").nunique("code").groupby("code").size().plot\
    (kind='bar', color="maroon", figsize=[20,7], fontsize=10, log=True, grid=False,
     title="number of different pdbs in each sequence similarity cluster", xlabel="number of unique protein sequences", ylabel="number of clusters")
b.figure.savefig(dest_path + "number_of_unique_pdbs_per_cluster.png")
b.clear()

xgb_joblib = joblib.load(arg_pars[0])
path_2_X_train = arg_pars[1]
path_2_y_train = arg_pars[2]
path_2_X_test = arg_pars[3]
path_2_y_test = arg_pars[4]
X_train = pickle.load(open(path_2_X_train, "rb"))
y_train = pickle.load(open(path_2_y_train, "rb"))
X_test = pickle.load(open(path_2_X_test, "rb"))
y_test = pickle.load(open(path_2_y_test, "rb"))


print(y_train)

y_train = y_train.values.astype(int)
y_test_for_decode = y_test.values.astype(int)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test_transformed = le.fit_transform(y_test_for_decode)

y_pred = xgb_joblib.predict(X_test)
y_prob = xgb_joblib.predict_proba(X_test)

result_dict = {}
result_dict["adjusted Balanced accuracy"] = round(metrics.balanced_accuracy_score(y_test_transformed, y_pred, adjusted=True), 3)
result_dict["f1_score"] = round(f1_score(y_test_transformed, y_pred, average='weighted'), 3)
result_dict["precision"] = round(precision_score(y_test_transformed, y_pred, average='weighted'), 3)
result_dict["recall"] = round(recall_score(y_test_transformed, y_pred, average='weighted'), 3)

#save dict to csv
with open(dest_path + "score_results.csv", 'w') as f:
    for key in result_dict:
        f.write(key + "," + str(result_dict[key]) + "\n")

conf_mat = metrics.confusion_matrix(y_test_transformed, y_pred)
conf_mat_df = pd.DataFrame(conf_mat)
# res = {}
# for cl in le.classes_:
#     res.update({cl:le.transform([cl])[0]})
# inv_map = {v: k for k, v in res.items()}

with open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/le_dict.pkl", 'rb') as f:
    le_dict = pickle.load(f)

inv_map = {v: k for k, v in le_dict.items()}

conf_mat_df.rename(inv_map, axis=1, inplace=True)
conf_mat_df.rename(inv_map, inplace=True)
pickle.dump(conf_mat_df, open(dest_path+"conf_mat_df.pkl", "wb"))
conf_mat_df_percent = conf_mat_df.div(conf_mat_df.sum(axis=1), axis=0).round(2)
s = sns.heatmap(conf_mat_df_percent, annot=True, fmt='g', cmap="BuPu")
s.set(xlabel='Prediction', ylabel='Actual lables')
s.figure.savefig(dest_path + "conf_mat_df_percent.png")
pickle.dump(conf_mat_df.sum(axis=1).to_dict(), open(dest_path+"actual_counts_per_qs.pkl", "wb"))

class_report = metrics.classification_report(y_test_transformed, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
class_report_df = class_report_df.round(3)
print(class_report_df)
class_report_df.index = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 24, "accuracy", "macro_avg", "weighted_avg"]
pickle.dump(class_report_df, open(dest_path+"class_report_df.pkl", "wb"))

# plot the confusion matrix using the probabilities
y_prob_df = pd.DataFrame(y_prob)
inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14, 11: 24}
y_prob_df.rename(columns=inv_map, inplace=True)

proba_pred_actual = pd.concat((y_prob_df.apply(lambda x: x.nlargest(3).index, axis=1, result_type='expand'),
                               y_test.reset_index().nsub.astype(int)), axis=1)
