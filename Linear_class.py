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
from preprocess_data import unique_players, unique_teams
from features import *
from matrix_fact import MF

#import data from 
merged_data = pd.read_json('./data/old_merged_data.json')

merged_data_2014 = merged_data[(merged_data['start_time'] < '2014-04-18') & (merged_data['start_time'] > '2013-10-01')]
merged_data_2015 = merged_data[(merged_data['start_time'] < '2015-04-18') & (merged_data['start_time'] > '2014-10-01')]
merged_data_2016 = merged_data[(merged_data['start_time'] < '2016-04-16') & (merged_data['start_time'] > '2015-10-01')]
merged_data_2017 = merged_data[(merged_data['start_time'] < '2017-04-15') & (merged_data['start_time'] > '2016-10-01')]
merged_data_2018 = merged_data[(merged_data['start_time'] < '2018-04-14') & (merged_data['start_time'] > '2017-10-01')]

playoff_data_2015 = merged_data[(merged_data['start_time'] > '2015-04-15') & (merged_data['start_time'] < '2015-10-01')]
playoff_data_2016 = merged_data[(merged_data['start_time'] > '2016-04-15') & (merged_data['start_time'] < '2016-10-01')]
playoff_data_2017 = merged_data[(merged_data['start_time'] > '2017-04-15') & (merged_data['start_time'] < '2017-10-01')]
playoff_data_2018 = merged_data[(merged_data['start_time'] > '2018-04-15') & (merged_data['start_time'] < '2018-10-01')]



merged_2015_2016_2017  = pd.concat([merged_data_2015,merged_data_2016, merged_data_2017])

merged_2015_2016_2017 = merged_2015_2016_2017.sample(frac=1).reset_index(drop=True)

#Create train, validation and test-sets
train_end = round(0.6*len(merged_data))
valid_end = train_end + (round(0.2*len(merged_data)))
train_index = range(0, train_end)
valid_index = range(train_end, valid_end)
test_index = range(valid_end, len(merged_data))

 
train_x = np.array(list(merged_data['player_array'][0:train_end]))
train_y = np.array(list(merged_data['win'][0:train_end]))
 
valid_x = np.array(list(merged_data['player_array'][train_end:valid_end]))
valid_y = np.array(list(merged_data['win'][train_end:valid_end]))

test_x = np.array(list(merged_data['player_array'][valid_end:len(merged_data)]))
test_y = np.array(list(merged_data['win'][valid_end:len(merged_data)]))


train_x = np.array(list(merged_2015_2016_2017['player_array']))
train_y = np.array(list(merged_2015_2016_2017['win']))
 
test_x = np.array(list(playoff_data_2018['player_array']))
test_y = np.array(list(playoff_data_2018['win']))

test_x = np.array(list(merged_data_2017['player_array']))
test_y = np.array(list(merged_data_2017['win']))

# Create Logistic regression object
regr = LogR(max_iter = 1000)

# Train the model using the training sets
regr.fit(train_x, train_y)

# Make predictions using the training set and test-set
train_pred = regr.predict(train_x)
test_pred = regr.predict(test_x)

print('train_accuracy_score:', accuracy_score(train_y, train_pred))    
print('test_accuracy_score:', accuracy_score(test_y, test_pred)) 



#create MLP object and train with complete data set
class_1 = MLPClassifier(solver='adam', hidden_layer_sizes = (40,) , activation = 'logistic', max_iter = 1000)
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
merged_data = field_goal(merged_data, unique_players())


train_field_x = np.array(list(merged_data['field_goal_array'][0:train_end]))
train_y = np.array(list(merged_data['win'][0:train_end])).reshape(train_end,1)
 
valid_field_x = np.array(list(merged_data['field_goal_array'][train_end:valid_end]))
valid_y = np.array(list(merged_data['win'][train_end:valid_end]))

test_field_x = np.array(list(merged_data['field_goal_array'][valid_end:len(merged_data)]))
test_y = np.array(list(merged_data['win'][valid_end:len(merged_data)]))


regr2 = LogR()
regr2.fit(train_field_x, train_y)

train_field_pred = regr2.predict(train_field_x)
test_field_pred = regr2.predict(test_field_x)

print('train_accuracy_score:', accuracy_score(train_y, train_field_pred))    
print('test_accuracy_score:', accuracy_score(test_y, test_field_pred)) 


class_field = MLPClassifier(solver='adam', hidden_layer_sizes = (40,), activation = 'logistic')
class_field.fit(train_field_x, train_y)

class_field.score(train_field_x, train_y)
class_field.score(test_field_x, test_y)

# Add feature Blocks  

merged_data = blocks(merged_data, unique_players())

train_blocks_x = np.concatenate((np.array(list(merged_data['player_array'][0:train_end])),
                                np.array(list(merged_data['blocks_array'][0:train_end]))), axis = 1)
train_y = np.array(list(merged_data['win'][0:train_end])).reshape(train_end,1)
 
valid_blocks_x = np.concatenate((np.array(list(merged_data['player_array'][train_end:valid_end])),
                                np.array(list(merged_data['blocks_array'][train_end:valid_end]))), axis = 1)
valid_y = np.array(list(merged_data['win'][train_end:valid_end])).reshape((valid_end - train_end),1)

test_blocks_x = np.concatenate((np.array(list(merged_data['player_array'][valid_end:len(merged_data)])),
                               np.array(list(merged_data['blocks_array'][valid_end:len(merged_data)]))), axis = 1)
test_y = np.array(list(merged_data['win'][valid_end:len(merged_data)])).reshape((len(merged_data) - valid_end),1)


regr3 = LogR()
regr3.fit(train_blocks_x, train_y)

train_blocks_pred = regr3.predict(train_blocks_x)
test_blocks_pred = regr3.predict(test_blocks_x)

print('train_accuracy_score:', accuracy_score(train_y, train_blocks_pred))    
print('test_accuracy_score:', accuracy_score(test_y, test_blocks_pred)) 


class_blocks = MLPClassifier(solver='adam', hidden_layer_sizes = (40,), activation = 'logistic')
class_blocks.fit(train_blocks_x, train_y)

class_blocks.score(train_blocks_x, train_y)
class_blocks.score(test_blocks_x, test_y)

# Add feature Assists  

merged_data = assists(merged_data, unique_players())

train_assists_x = np.concatenate((np.array(list(merged_data['player_array'][0:train_end])),
                                np.array(list(merged_data['assists_array'][0:train_end]))), axis = 1)
train_y = np.array(list(merged_data['win'][0:train_end])).reshape(train_end,1)
 
valid_assists_x = np.concatenate((np.array(list(merged_data['player_array'][train_end:valid_end])),
                                np.array(list(merged_data['assists_array'][train_end:valid_end]))), axis = 1)
valid_y = np.array(list(merged_data['win'][train_end:valid_end])).reshape((valid_end - train_end),1)

test_assists_x = np.concatenate((np.array(list(merged_data['player_array'][valid_end:len(merged_data)])),
                               np.array(list(merged_data['assists_array'][valid_end:len(merged_data)]))), axis = 1)
test_y = np.array(list(merged_data['win'][valid_end:len(merged_data)])).reshape((len(merged_data) - valid_end),1)


regr4 = LogR()
regr4.fit(train_assists_x, train_y)

train_assists_pred = regr4.predict(train_assists_x)
test_assists_pred = regr4.predict(test_assists_x)

print('train_accuracy_score:', accuracy_score(train_y, train_assists_pred))    
print('test_accuracy_score:', accuracy_score(test_y, test_assists_pred)) 


class_assists = MLPClassifier(solver='adam', hidden_layer_sizes = (40,), activation = 'logistic')
class_assists.fit(train_assists_x, train_y)

class_assists.score(train_assists_x, train_y)
class_assists.score(test_assists_x, test_y)



# Add feature minutes_played  

merged_data = minutes_played(merged_data, unique_players())

train_minutes_played_x = np.concatenate((np.array(list(merged_data['player_array'][0:train_end])),
                                np.array(list(merged_data['minutes_played_array'][0:train_end]))), axis = 1)
train_y = np.array(list(merged_data['win'][0:train_end])).reshape(train_end,1)
 
valid_minutes_played_x = np.concatenate((np.array(list(merged_data['player_array'][train_end:valid_end])),
                                np.array(list(merged_data['minutes_played_array'][train_end:valid_end]))), axis = 1)
valid_y = np.array(list(merged_data['win'][train_end:valid_end])).reshape((valid_end - train_end),1)

test_minutes_played_x = np.concatenate((np.array(list(merged_data['player_array'][valid_end:len(merged_data)])),
                               np.array(list(merged_data['minutes_played_array'][valid_end:len(merged_data)]))), axis = 1)
test_y = np.array(list(merged_data['win'][valid_end:len(merged_data)])).reshape((len(merged_data) - valid_end),1)


regr5 = LogR()
regr5.fit(train_minutes_played_x, train_y)

train_minutes_played_pred = regr5.predict(train_minutes_played_x)
test_minutes_played_pred = regr5.predict(test_minutes_played_x)

print('train_accuracy_score:', accuracy_score(train_y, train_minutes_played_pred))    
print('test_accuracy_score:', accuracy_score(test_y, test_minutes_played_pred)) 


class_minutes_played = MLPClassifier(solver='adam', hidden_layer_sizes = (40,), activation = 'logistic')
class_minutes_played.fit(train_minutes_played_x, train_y)

class_minutes_played.score(train_minutes_played_x, train_y)
class_minutes_played.score(test_minutes_played_x, test_y)

# Add feature points  

merged_data = points(merged_data, unique_players())

train_points_x = np.concatenate((np.array(list(merged_data['player_array'][0:train_end])),
                                np.array(list(merged_data['points_array'][0:train_end]))), axis = 1)
train_y = np.array(list(merged_data['win'][0:train_end])).reshape(train_end,1)
 
valid_points_x = np.concatenate((np.array(list(merged_data['player_array'][train_end:valid_end])),
                                np.array(list(merged_data['points_array'][train_end:valid_end]))), axis = 1)
valid_y = np.array(list(merged_data['win'][train_end:valid_end])).reshape((valid_end - train_end),1)

test_points_x = np.concatenate((np.array(list(merged_data['player_array'][valid_end:len(merged_data)])),
                               np.array(list(merged_data['points_array'][valid_end:len(merged_data)]))), axis = 1)
test_y = np.array(list(merged_data['win'][valid_end:len(merged_data)])).reshape((len(merged_data) - valid_end),1)


regr6 = LogR()
regr6.fit(train_points_x, train_y)

train_points_pred = regr6.predict(train_points_x)
test_points_pred = regr6.predict(test_points_x)



print('train_accuracy_score:', accuracy_score(train_y, train_points_pred))    
print('test_accuracy_score:', accuracy_score(test_y, test_points_pred)) 


class_points = MLPClassifier(solver='adam', hidden_layer_sizes = (40,), activation = 'logistic')
class_points.fit(train_points_x, train_y)

class_points.score(train_points_x, train_y)
class_points.score(test_points_x, test_y)

# Add feature efg  

merged_data = efg(merged_data, unique_players())

train_efg_x = np.array(list(merged_data['efg_array'][0:train_end]))
train_y = np.array(list(merged_data['win'][0:train_end]))
 
valid_efg_x = np.array(list(merged_data['efg_array'][train_end:valid_end]))
valid_y = np.array(list(merged_data['win'][train_end:valid_end]))

test_efg_x = np.array(list(merged_data['efg_array'][valid_end:len(merged_data)]))
test_y = np.array(list(merged_data['win'][valid_end:len(merged_data)]))


regr7 = LogR(max_iter = 500)
regr7.fit(train_efg_x, train_y)

train_efg_pred = regr7.predict(train_efg_x)
test_efg_pred = regr7.predict(test_efg_x)

print('train_accuracy_score:', accuracy_score(train_y, train_efg_pred))    
print('test_accuracy_score:', accuracy_score(test_y, test_efg_pred)) 


class_efg = MLPClassifier(solver='adam', hidden_layer_sizes = (40,), activation = 'logistic')
class_efg.fit(train_efg_x, train_y)

class_efg.score(train_efg_x, train_y)
class_efg.score(test_efg_x, test_y)


# Add feature stats to make a 3d array 

merged_data = player_stats(merged_data, unique_players())

train_stats_x = np.array(list(merged_data['stats_array'][0:train_end]))
train_y = np.array(list(merged_data['win'][0:train_end]))

valid_stats_x = np.array(list(merged_data['stats_array'][train_end:valid_end]))
valid_y = np.array(list(merged_data['win'][train_end:valid_end])).reshape((valid_end - train_end),1)

test_stats_x = np.array(list(merged_data['stats_array'][valid_end:len(merged_data)]))
test_y = np.array(list(merged_data['win'][valid_end:len(merged_data)]))


regr7 = LogR(max_iter=100)
regr7.fit(train_stats_x, train_y)

train_stats_pred = regr7.predict(train_stats_x)
test_stats_pred = regr7.predict(test_stats_x)

print('train_accuracy_score:', accuracy_score(train_y, train_stats_pred))    
print('test_accuracy_score:', accuracy_score(test_y, test_stats_pred)) 


class_stats = MLPClassifier(solver='adam', hidden_layer_sizes = (40,), activation = 'logistic')
class_stats.fit(train_stats_x, train_y)

class_stats.score(train_stats_x, train_y)
class_stats.score(test_stats_x, test_y)

#convert back to 2D
for i in range(len(merged_data)):
    merged_data['stats_array'][i] = merged_data['stats_array'][i].reshape(7440)


R = np.array([
    [(2,5), (2,5), (2,5)],
    [(2,5), (2,5), (2,5)],
    [(2,5), (2,5),(2,5)],
    [(2,5), (2,5),(2,5)]
])

mf = MF(R, K=6, alpha=0.1, beta=0.01, iterations=20)
 
training_process = mf.train()
print(mf.P)
print(mf.Q)
print(mf.full_matrix())
   



W = np.array([
        [[(4,5),5,6,8],[1,5,6,8],[1,5,6,8],[1,5,6,8]],
        [[1,0,0,0],[1,2,3,4],[1,45,88,0],[1,0,0,0]],
        [[1,5,6,8],[1,5,6,8],[1,5,6,8],[1,5,6,8]],
        [[1,5,6,8],[1,5,6,8],[1,5,6,8],[1,5,6,8]]
        ])
    
    

    
 
