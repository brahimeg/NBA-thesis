# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:44:59 2020

@author: Ibrahim-main
"""
#####	DISCLAIMER
##### Matrix factorization algorithms was not written by myself
#####	DISCLAIMER	

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
from preprocess_data import unique_players, process_playerdata
from features import *
from matrix_fact import MF
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

#import data from 
merged_data = pd.read_json('./data/merged_data.json')

merged_data_2014 = merged_data[(merged_data['start_time'] < '2014-04-18') & (merged_data['start_time'] > '2013-10-01')]
merged_data_2015 = merged_data[(merged_data['start_time'] < '2015-04-18') & (merged_data['start_time'] > '2014-10-01')]
merged_data_2016 = merged_data[(merged_data['start_time'] < '2016-04-16') & (merged_data['start_time'] > '2015-10-01')]
merged_data_2017 = merged_data[(merged_data['start_time'] < '2017-04-15') & (merged_data['start_time'] > '2016-10-01')]
merged_data_2018 = merged_data[(merged_data['start_time'] < '2018-04-14') & (merged_data['start_time'] > '2017-10-01')]

playoff_data_2015 = merged_data[(merged_data['start_time'] > '2015-04-15') & (merged_data['start_time'] < '2015-10-01')]
playoff_data_2016 = merged_data[(merged_data['start_time'] > '2016-04-15') & (merged_data['start_time'] < '2016-10-01')]
playoff_data_2017 = merged_data[(merged_data['start_time'] > '2017-04-15') & (merged_data['start_time'] < '2017-10-01')]
playoff_data_2018 = merged_data[(merged_data['start_time'] > '2018-04-15') & (merged_data['start_time'] < '2018-10-01')]


def unique_teams(year):
    player_data = process_playerdata()
    uniq_teams= {None}
    set(uniq_teams)
    for x in player_data[year].keys():
        uniq_teams.add(player_data[year][x]['team'])
    uniq_teams.remove(None)
    return uniq_teams


uniq_teams_2014 = list(unique_teams('2014'))
uniq_teams_2015 = list(unique_teams('2015'))
uniq_teams_2016 = list(unique_teams('2016'))
uniq_teams_2017 = list(unique_teams('2017'))
uniq_teams_2018 = list(unique_teams('2018'))

player_data = process_playerdata()

#assists, 2014, merged_data_2014
mat_2014 = {}
for i in range(len(merged_data_2014)):
    if merged_data_2014['home_team'][i] not in mat_2014.keys() and merged_data_2014['away_team'][i] not in mat_2014.keys():
        mat_2014[merged_data_2014['home_team'][i]] = pd.DataFrame(index=merged_data_2014['home_players'][i].keys(),
                columns=merged_data_2014['home_players'][i].keys())               

        mat_2014[merged_data_2014['away_team'][i]] = pd.DataFrame(index=merged_data_2014['away_players'][i].keys(),
                columns=merged_data_2014['away_players'][i].keys())
        
for team in uniq_teams_2014:
    vals_array = pd.Series(np.zeros(len(mat_2014[team])), mat_2014[team].index)
    for player in vals_array.index:
        try:
            vals_array[player] = player_data['2013'][player]['assists']
        except KeyError:
            vals_array[player] = 0
    for x in mat_2014[team].index:
        for y in mat_2014[team].columns:
            if x == y:
                mat_2014[team].loc[x,y] = 1
            else:
                try:
                    mat_2014[team].loc[x,y] = round(player_data['2013'][x]['assists'] / player_data['2013'][y]['assists'], 3)
                except:
                    mat_2014[team].loc[x,y] = 0
    mat_2014[team] = mat_2014[team].replace([np.inf, -np.inf], 0)       
                
            

    


V = np.array(mat_2014['MIAMI HEAT'])

N = len(V)
M = len(V)
K = 10

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ = matrix_factorization(V, P, Q, K)
nR = np.dot(nP, nQ.T)



MF_SGD = ExplicitMF(V, 40, learning='sgd', verbose=True)

print(MF_SGD)

V = np.array(mat_2014['MIAMI HEAT'])

norm =  V/ np.linalg.norm(V)


mf = MF(norm, K=6, alpha=0.1, beta=0.01, iterations=200)
 
training_process = mf.train()
print(mf.P)
print(mf.Q)
print(mf.full_matrix())
test = mf.full_matrix() * np.linalg.norm(R)




def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T


player_data = process_playerdata()

a =  merged_data['player_array_away'][4000]

counter = 0 
for x in merged_data['efg_array'][6482]:
    if x != 0:
        counter = counter  + 1


for x in merged_data['away_players'][4000]:
    print(x)


















 
