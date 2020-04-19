# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:56:36 2020

@author: Ibrahim-main
"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import sklearn as sk
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import classification_report, recall_score, f1_score, roc_auc_score
from mlxtend.evaluate import mcnemar, mcnemar_table
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import itertools
#import tensorflow as tf


def run_test(trainData, testData, algo, combinations=None):
    features  = ['player_array', 'usage_percentage_array', 'PER_array', 'win_shares_array', 'blocks_array', 'assists_array', 'minutes_played_array',
                 'points_array', 'efg_array']
    if combinations == None:       
        combinations = []
        for i in range(len(features)):
            combinations = list(itertools.combinations(features, i+1)) + combinations
        
    scores = {}
    counter = 0
    for item in combinations:
        counter += 1
        print(item, counter)
        trainData['temp'] = None
        testData['temp'] = None
        train = combine_features(trainData, item)
        test = combine_features(testData, item)
        for i in trainData.index:        
            trainData['temp'][i] = train[i-trainData.index[0]]
        for i in testData.index:        
            testData['temp'][i] = test[i-testData.index[0]]
        scores[item] = test_function(trainData, testData, algo, ['temp'])
    return scores

def top_5(scores):
    scores = {k: v for k, v in sorted(scores.items(),reverse=True, key=lambda item: item[1][1])}
    return list(scores.items())[0:5]

def combine_features(data, feature_list):
    print(len(feature_list))
    output = np.array(list(data[feature_list[0]]))
    if len(feature_list) ==  1:
        return output
    else:
        for i in range(len(feature_list)-1):
            output = np.concatenate((output, np.array(list(data[feature_list[i+1]]))), axis = 1)
        return output


def log_p_value(trainData,testData, input_pred):
    log = LogR(max_iter=500, solver='newton-cg', C=0.1)
    
    train_x = np.array(list(trainData['player_array']))
    train_y = np.array(list(trainData['win']))
    test_x = np.array(list(testData['player_array']))
    test_y = np.array(list(testData['win']))
    
    
    log.fit(train_x, train_y)

    test_pred = log.predict(test_x)
    
    tb = mcnemar_table(y_target=test_y, y_model1=input_pred, y_model2=test_pred)
    chi2, p = mcnemar(ary=tb, corrected=True)
           
    return p

def mlp_p_value(trainData,testData, input_pred):
    ann = MLPClassifier(verbose=True, max_iter= 500,tol= 0.0005, solver= 'adam', alpha= 0.0001, activation= 'logistic', hidden_layer_sizes = (50,40))
    train_x = np.array(list(trainData['player_array']))
    train_y = np.array(list(trainData['win']))
    test_x = np.array(list(testData['player_array']))
    test_y = np.array(list(testData['win']))
    
    ann.fit(train_x, train_y)

    test_pred = ann.predict(test_x)
    
    tb = mcnemar_table(y_target=test_y, y_model1=input_pred, y_model2=test_pred)
    chi2, p = mcnemar(ary=tb, corrected=True)
           
    return p

def svm_p_value(trainData,testData, input_pred):
    svc = LinearSVC(max_iter = 10000, verbose=50, C= 0.1)
    train_x = np.array(list(trainData['player_array']))
    train_y = np.array(list(trainData['win']))
    test_x = np.array(list(testData['player_array']))
    test_y = np.array(list(testData['win']))
    
    svc.fit(train_x, train_y)

    test_pred = svc.predict(test_x)
    
    tb = mcnemar_table(y_target=test_y, y_model1=input_pred, y_model2=test_pred)
    chi2, p = mcnemar(ary=tb, corrected=True)
           
    return p

def test_function(trainData, testData, algorithm, playerstats = ['player_array']):
    results = open('results.txt', 'a')
    start = trainData.iloc[0,0].year
    end = trainData.iloc[len(trainData)-1,0].year
    test_start = testData.iloc[0,0].year
    test_end = testData.iloc[len(testData)-1,0].year
    results.write('### Training on {}/{} season(s) and Testing on {}/{} season \n'.format(start, end, test_start, test_end))
    
        
    if algorithm[0] == "log":      
        for x in playerstats:
            results.write('Using ' + x + '\n')
            #parameter_space = {
            #                    'C': [1,0.5,0.1],
            #                    'solver' : ['newton-cg', 'lbfgs', 'sag', 'liblinear' ]
            #                    }
            regr = algorithm[1]
            train_x = np.array(list(trainData[x]))
            train_y = np.array(list(trainData['win']))
            test_x = np.array(list(testData[x]))
            test_y = np.array(list(testData['win']))     
            #regr = GridSearchCV(log, parameter_space, n_jobs=-1, cv=5, verbose=50)
            regr.fit(train_x, train_y)
            
            train_pred = regr.predict(train_x)
            test_pred = regr.predict(test_x)
            
            results.write('train_accuracy_score: '+ str(accuracy_score(train_y, train_pred)) + '\n')
            results.write('test_accuracy_score: '+ str(accuracy_score(test_y, test_pred)) + '\n\n')
            results.write('train_recall_score: '+ str(recall_score(train_y, train_pred)) + '\n')
            results.write('test_recall_score: '+ str(recall_score(test_y, test_pred)) + '\n\n')
            results.write('train_f1_score: '+ str(f1_score(train_y, train_pred)) + '\n')
            results.write('test_f1_score: '+ str(f1_score(test_y, test_pred)) + '\n\n')
            results.write('train_roc_auc_score: '+ str(roc_auc_score(train_y, train_pred)) + '\n')
            results.write('test_roc_auc_score: '+ str(roc_auc_score(test_y, test_pred)) + '\n\n')
            
            p = log_p_value(trainData, testData, test_pred)
            
            return accuracy_score(train_y, train_pred), accuracy_score(test_y, test_pred), regr, p

    if algorithm[0] == "MLP":
        for x in playerstats:
            print(x)
            results.write('Using ' + x + '\n')
#            parameter_space = {
#                'hidden_layer_sizes': [(100,100),(100,80),(100,50),(100,20),(50,50),(50,40),(50,20),(100,),(50,)],
#                'alpha': [0.0001,0.001],
#                'activation': ['logistic', 'identity', 'tanh']
#            }
            ann = algorithm[1]
            train_x = np.array(list(trainData[x]))
            train_y = np.array(list(trainData['win']))
            test_x = np.array(list(testData[x]))
            test_y = np.array(list(testData['win']))    
            #ann = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, verbose=50)
            ann.fit(train_x, train_y)
            
            train_pred = ann.predict(train_x)
            test_pred = ann.predict(test_x)
            
            results.write('train_accuracy_score: '+ str(accuracy_score(train_y, train_pred)) + '\n')
            results.write('test_accuracy_score: '+ str(accuracy_score(test_y, test_pred)) + '\n\n')
            results.write('train_recall_score: '+ str(recall_score(train_y, train_pred)) + '\n')
            results.write('test_recall_score: '+ str(recall_score(test_y, test_pred)) + '\n\n')
            results.write('train_f1_score: '+ str(f1_score(train_y, train_pred)) + '\n')
            results.write('test_f1_score: '+ str(f1_score(test_y, test_pred)) + '\n\n')
            results.write('train_roc_auc_score: '+ str(roc_auc_score(train_y, train_pred)) + '\n')
            results.write('test_roc_auc_score: '+ str(roc_auc_score(test_y, test_pred)) + '\n\n')
            
            p = mlp_p_value(trainData, testData, test_pred)
            
            return ann.score(train_x, train_y), ann.score(test_x, test_y), ann, p
    a
    if algorithm[0]== "SVM":
        for x in playerstats:
            print(x)
            results.write('Using ' + x + '\n')
#            parameter_space = {
#                'C': [0.1,0.3,0.7,1,1.2,1.5]
#            }
            svc = algorithm[1]
            train_x = np.array(list(trainData[x]))
            train_y = np.array(list(trainData['win']))
            test_x = np.array(list(testData[x]))
            test_y = np.array(list(testData['win']))    
#            svc = GridSearchCV(test, parameter_space, n_jobs=-1, cv=3, verbose=50)
            svc.fit(train_x, train_y)
            
            train_pred = svc.predict(train_x)
            test_pred = svc.predict(test_x)
            
            results.write('train_accuracy_score: '+ str(accuracy_score(train_y, train_pred)) + '\n')
            results.write('test_accuracy_score: '+ str(accuracy_score(test_y, test_pred)) + '\n\n')
            results.write('train_recall_score: '+ str(recall_score(train_y, train_pred)) + '\n')
            results.write('test_recall_score: '+ str(recall_score(test_y, test_pred)) + '\n\n')
            results.write('train_f1_score: '+ str(f1_score(train_y, train_pred)) + '\n')
            results.write('test_f1_score: '+ str(f1_score(test_y, test_pred)) + '\n\n')
            results.write('train_roc_auc_score: '+ str(roc_auc_score(train_y, train_pred)) + '\n')
            results.write('test_roc_auc_score: '+ str(roc_auc_score(test_y, test_pred)) + '\n\n')
            
            p = svm_p_value(trainData, testData, test_pred)
            
            return svc.score(train_x, train_y), svc.score(test_x, test_y), svc, p

def normalize(merged_data):
    features  = ['usage_percentage_array', 'PER_array', 'win_shares_array', 'blocks_array', 'assists_array', 'minutes_played_array',
                 'points_array', 'efg_array']

    temp_dict= {}
    for item in features:
        print(item)
        temp_dict[item] = []
        for i in range(len(merged_data)):
            for x in merged_data[item][i]:
                if x != 0:
                    temp_dict[item].append(x)
        temp_dict[item] = min(temp_dict[item]), max(temp_dict[item])
    
    for item in features:
        print(item)
        for i in range(len(merged_data)):
            for j, element in enumerate(merged_data[item][i]):
                if element != 0:
                    merged_data[item][i][j] = (element - temp_dict[item][0])/(temp_dict[item][1] - temp_dict[item][0])
                    
    return merged_data

           
