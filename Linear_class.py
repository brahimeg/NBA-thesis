# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:13:32 2020

@author: Ibrahim
"""
import pandas as pd 
import numpy as np
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType
import os
import glob
import datetime
from datetime import timedelta
import sklearn as sk
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from preprocess_data import unique_players

#import data from 
merged_data = pd.read_json('./data/merged_data.json')

merged_data = merged_data.sample(frac=1).reset_index(drop=True)

#Create train, validation and test-sets
train_end = round(0.6*len(merged_data))
valid_end = train_end + (round(0.2*len(merged_data)))
train_index = range(0, train_end)
valid_index = range(train_end, valid_end)
test_index = range(valid_end, len(merged_data))
 
 
train_x = np.array(list(merged_data['player_array'][0:train_end]))
train_y = np.array(list(merged_data['win'][0:train_end])).reshape(train_end,1)
 
valid_x = np.array(list(merged_data['player_array'][train_end:valid_end]))
valid_y = np.array(list(merged_data['win'][train_end:valid_end])).reshape((valid_end - train_end),1)

test_x = np.array(list(merged_data['player_array'][valid_end:len(merged_data)]))
test_y = np.array(list(merged_data['win'][valid_end:len(merged_data)])).reshape((len(merged_data) - valid_end),1)

# Create Logistic regression object
regr = LogR()

# Train the model using the training sets
regr.fit(train_x, train_y)

# Make predictions using the training set and test-set
train_pred = regr.predict(train_x)
test_pred = regr.predict(test_x)

print('train_accuracy_score:', accuracy_score(train_y, train_pred))    
print('test_accuracy_score:', accuracy_score(test_y, test_pred)) 



#create MLP object and train with complete data set
class_1 = MLPClassifier(solver='adam', hidden_layer_sizes = (40,) , activation = 'logistic')
class_1.fit(train_x, train_y)

class_1.score(train_x, train_y)
class_1.score(test_x, test_y)


#Instead of binary classification of winning use regression of point difference to estimate winner

merged_data['point_diff'] = merged_data['home_team_score'] - merged_data['away_team_score']

train1_x = np.array(list(merged_data['player_array'][0:train_end]))
train1_y = np.array(list(merged_data['point_diff'][0:train_end])).reshape(train_end,1)

test1_x = np.array(list(merged_data['player_array'][valid_end:len(merged_data)]))
test1_y = np.array(list(merged_data['point_diff'][valid_end:len(merged_data)])).reshape((len(merged_data) - valid_end),1)


#create LR and MLP object and fit on training set ussing point-diff as y-labels instead of binaray-win-labels
regr1 = LR()
regr1.fit(train1_x, train1_y)


train1_pred = regr1.predict(train1_x)
train1_pred = train1_pred > 0
train1_pred = train1_pred.astype(int)

test1_pred = regr1.predict(test1_x)
test1_pred = test1_pred > 0
test1_pred = test1_pred.astype(int)


print('train_accuracy_score:', accuracy_score(train_y, train1_pred))
print('test_accuracy_score:', accuracy_score(test_y, test1_pred)) 


class_2 = MLPClassifier(solver='adam', hidden_layer_sizes = (40,) , activation = 'logistic')
class_2.fit(train1_x, train1_y)

a = class_2.predict(train1_x)
a = a > 0
a = a.astype(int)

b = class_2.predict(test1_x)
b = b > 0
b = b.astype(int)

print('train_accuracy_score:', accuracy_score(train_y, a))
print('test_accuracy_score:', accuracy_score(test_y, b))


#Use two estimators (player_array and field-goal percentage) to predict win-probability

uniq_players = unique_players()

merged_data['field_goal_array_home'] = None
merged_data['field_goal_array_away'] = None
merged_data['field_goal_array'] = None
    
for i in range(len(merged_data)):
    print(i)
    temp_home = {}
    temp_away = {}
    for x in uniq_players:
        temp_home[x] = 0
        temp_away[x] = 0
        for y in merged_data['players'][i].keys():
            if y == x:
                if merged_data['players'][i][y]['team'] == merged_data['home_team'][i]:
                    try:
                        temp_home[x] = merged_data['players'][i][y]['made_field_goals'] / merged_data['players'][i][y]['attempted_field_goals']
                    except:
                        temp_home[x] = 0
                elif merged_data['players'][i][y]['team'] == merged_data['away_team'][i]:
                    try:
                        temp_away[x] = merged_data['players'][i][y]['made_field_goals'] / merged_data['players'][i][y]['attempted_field_goals']
                    except:
                        temp_away[x] = 0
    merged_data['field_goal_array_home'][i] = np.array(list(temp_home.values()))
    merged_data['field_goal_array_away'][i] = np.array(list(temp_away.values()))
    merged_data['field_goal_array'][i] = np.concatenate((merged_data['field_goal_array_home'][i], merged_data['field_goal_array_away'][i]))



train_x = np.array(list(merged_data['player_array'][0:train_end]))
train_field_x = np.array(list(merged_data['field_goal_array'][0:train_end]))
train_y = np.array(list(merged_data['win'][0:train_end])).reshape(train_end,1)
 
valid_x = np.array(list(merged_data['player_array'][train_end:valid_end]))
valid_field_x = np.array(list(merged_data['field_goal_array'][train_end:valid_end]))
valid_y = np.array(list(merged_data['win'][train_end:valid_end])).reshape((valid_end - train_end),1)

test_x = np.array(list(merged_data['player_array'][valid_end:len(merged_data)]))
test_field_x = np.array(list(merged_data['field_goal_array'][valid_end:len(merged_data)]))
test_y = np.array(list(merged_data['win'][valid_end:len(merged_data)])).reshape((len(merged_data) - valid_end),1)



regr2 = LogR()
regr2.fit(np.concatenate((train_x,train_field_x), axis = 1), train_y)

train_field_pred = regr2.predict(np.concatenate((train_x,train_field_x), axis = 1))
test_field_pred = regr2.predict(np.concatenate((test_x,test_field_x), axis = 1))

print('train_accuracy_score:', accuracy_score(train_y, train_field_pred))    
print('test_accuracy_score:', accuracy_score(test_y, test_field_pred)) 




    