# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:13:32 2020

@author: Ibrahim
"""
import pandas as pd 
import numpy as np
import researchpy as rp
from mlxtend.evaluate import mcnemar, mcnemar_table
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType
import os
import glob
import datetime
import matplotlib
import matplotlib.pyplot as plt
from datetime import timedelta
import sklearn as sk
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.linear_model import LogisticRegressionCV as LogCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from preprocess_data import unique_players, process_playerdata
from sklearn.feature_selection import SelectFromModel, RFE
from features import *
from Testing import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, recall_score, f1_score, roc_auc_score
from sklearn import *


#import data from 
merged_data = pd.read_json('./data/big_merged_data.json')
#merged_data = pd.read_json('./data/before_most/old_merged_data.json')
#merged_data = pd.read_json('./data/merged_data.json')
merged_data = normalize(merged_data)

merged_data_2014 = merged_data[(merged_data['start_time'] < '2014-04-18') & (merged_data['start_time'] > '2013-10-01')]
merged_data_2015 = merged_data[(merged_data['start_time'] < '2015-04-18') & (merged_data['start_time'] > '2014-10-01')]
merged_data_2016 = merged_data[(merged_data['start_time'] < '2016-04-16') & (merged_data['start_time'] > '2015-10-01')]
merged_data_2017 = merged_data[(merged_data['start_time'] < '2017-04-15') & (merged_data['start_time'] > '2016-10-01')]
merged_data_2018 = merged_data[(merged_data['start_time'] < '2018-04-14') & (merged_data['start_time'] > '2017-10-01')]

playoff_data_2014 = merged_data[(merged_data['start_time'] > '2014-04-15') & (merged_data['start_time'] < '2014-10-01')]
playoff_data_2015 = merged_data[(merged_data['start_time'] > '2015-04-15') & (merged_data['start_time'] < '2015-10-01')]
playoff_data_2016 = merged_data[(merged_data['start_time'] > '2016-04-15') & (merged_data['start_time'] < '2016-10-01')]
playoff_data_2017 = merged_data[(merged_data['start_time'] > '2017-04-15') & (merged_data['start_time'] < '2017-10-01')]
playoff_data_2018 = merged_data[(merged_data['start_time'] > '2018-04-15') & (merged_data['start_time'] < '2018-10-01')]

#merged_data = merged_data.sample(frac=1).reset_index(drop=True)
merged_2014_2015_2016_2017  = pd.concat([merged_data_2014, merged_data_2015,merged_data_2016, merged_data_2017]).reset_index(drop=True)
merged_2014_2015_2016  = pd.concat([merged_data_2014,merged_data_2015,merged_data_2016]).reset_index(drop=True)
merged_2014_2015 = pd.concat([merged_data_2014,merged_data_2015]).reset_index(drop=True)
merged_2015_2016 = pd.concat([merged_data_2015,merged_data_2016]).reset_index(drop=True)

    
##Methods for generating stats-arrays
#merged_data = usage_percentage(merged_data, unique_players())
#merged_data = PER(merged_data, unique_players())
#merged_data = win_shares(merged_data, unique_players())
#merged_data = blocks(merged_data, unique_players())
#merged_data = assists(merged_data, unique_players())
#merged_data = minutes_played(merged_data, unique_players())
#merged_data = points(merged_data, unique_players())
#merged_data = efg(merged_data, unique_players())



logarithms = ['log', 'MLP', 'SVM']

score_log = run_test(merged_data_2015, merged_data_2016,algo=tuple(('log', log)))    
top5_log = dict(top_5(score_log))

score_ann = run_test(merged_data_2016, merged_data_2017, algo = tuple(('MLP', ann)), 
                     combinations = (list(top5_log0)+list(top5_log1)+list(top5_log2)) )
top5_ann = dict(top_5(score_ann))

score_svm = run_test(merged_data_2015, merged_data_2016, algo = 'SVM', combinations=top5_log.keys())
top5_svm = dict(top_5(score_svm))

test = run_test(merged_data_2015, merged_data_2016, algo = tuple(('log',log)), combinations = [best_combis[0]])
p = test[('usage_percentage_array', 'win_shares_array', 'assists_array')][3]
p


#### LOG

log = LogR(max_iter=500, solver='newton-cg', C=0.1)
train_x = np.array(list(merged_data_2014['player_array']))
train_y = np.array(list(merged_data_2014['win']))
test_x = np.array(list(merged_data_2015['player_array']))
test_y = np.array(list(merged_data_2015['win']))       

#parameter_space = {
#                    'C': [1,0.5],
#                    'solver' : ['newton-cg', 'lbfgs', ]
#                    }
#clf = GridSearchCV(log, parameter_space, n_jobs=-1, verbose=50)
log.fit(train_x, train_y)

train_pred = log.predict(train_x)
test_pred = log.predict(test_x)

accuracy_score(train_y, train_pred)    
accuracy_score(test_y, test_pred)

recall_score(train_y, train_pred)    
recall_score(test_y, test_pred)

f1_score(train_y, train_pred)    
f1_score(test_y, test_pred)

roc_auc_score(train_y, train_pred)
roc_auc_score(test_y, test_pred)


#####  MLP

ann = MLPClassifier(verbose=True, max_iter= 500,tol= 0.0005, solver= 'adam', alpha= 0.0001, activation= 'logistic', hidden_layer_sizes = (50,40))
train_x = np.array(list(merged_2014_2015_2016_2017['player_array']))
train_y = np.array(list(merged_2014_2015_2016_2017['win']))
test_x = np.array(list(merged_data_2018['player_array']))
test_y = np.array(list(merged_data_2018['win']))       

ann.fit(train_x, train_y)

train_pred = ann.predict(train_x)
test_pred = ann.predict(test_x)

accuracy_score(train_y, train_pred)    
accuracy_score(test_y, test_pred)

recall_score(train_y, train_pred)    
recall_score(test_y, test_pred)

f1_score(train_y, train_pred)    
f1_score(test_y, test_pred)

roc_auc_score(train_y, train_pred)
roc_auc_score(test_y, test_pred)



#####  SVC
svc = LinearSVC(max_iter = 10000, verbose=50, C= 0.1)
#svc = svm.SVC()
train_x = np.array(list(merged_data_2015['player_array']))
train_y = np.array(list(merged_data_2015['win']))
test_x = np.array(list(merged_data_2016['player_array']))
test_y = np.array(list(merged_data_2016['win']))       

svc.fit(train_x, train_y)

train_pred = svc.predict(train_x)
test_pred = svc.predict(test_x)

accuracy_score(train_y, train_pred)    
accuracy_score(test_y, test_pred)

recall_score(train_y, train_pred)    
recall_score(test_y, test_pred)

f1_score(train_y, train_pred)    
f1_score(test_y, test_pred)

roc_auc_score(train_y, train_pred)
roc_auc_score(test_y, test_pred)


##result plots


labels = ['2014/2015', '2015/2016', '2016/2017', '2017/2018']
log_vals = [0.606, 0.659, 0.623, 0.613]
mlp_vals = [0.617, 0.650, 0.628, 0.626]
svm_vals = [0.616, 0.658, 0.625, 0.621]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - (width/2), log_vals, width, label='LOG')
rects2 = ax.bar(x + (width/2), mlp_vals, width, label='ANN')
rects3 = ax.bar((x - (width/2)) - width, svm_vals, width, label='SVM')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right')
ax.set_ylim([0.50,0.68])

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)

fig.tight_layout()

plt.show()
fig.savefig('fig1.png', dpi=500)



labels = ['2015/2016', '2016/2017', '2017/2018']
log_vals = [0.659, 0.631, 0.599]
mlp_vals = [0.666, 0.621, 0.594]
svm_vals = [0.663, 0.633, 0.588]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - (width/2), log_vals, width, label='LOG')
rects2 = ax.bar(x + (width/2), mlp_vals, width, label='ANN')
rects3 = ax.bar((x - (width/2)) - width, svm_vals, width, label='SVM')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right')
ax.set_ylim([0.5,0.68])


#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)

fig.tight_layout()

plt.show()
fig.savefig('fig2.png', dpi=500)



# Example of calculating the mcnemar test
from statsmodels.stats.contingency_tables import mcnemar
# define contingency table
table = [[4, 2],
		 [1, 3]]
# calculate mcnemar test
result = mcnemar(table, correction=False)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')

