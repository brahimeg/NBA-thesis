# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:39:48 2020

@author: Ibrahim-main
"""
import pandas as pd 
import numpy as np
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType
import os
import glob
import datetime
from datetime import timedelta
import sklearn as ska 

def usage_percentage(merged_data, uniq_players):
    merged_data['usage_percentage_array_home'] = None
    merged_data['usage_percentage_array_away'] = None
    merged_data['usage_percentage_array'] = None
        
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
                            temp_home[x] = merged_data['players'][i][y]['usage_percentage'] 
                        except:
                            temp_home[x] = 0
                    elif merged_data['players'][i][y]['team'] == merged_data['away_team'][i]:
                        try:
                            temp_away[x] = merged_data['players'][i][y]['usage_percentage']
                        except:
                            temp_away[x] = 0
        merged_data['usage_percentage_array_home'][i] = np.array(list(temp_home.values()))
        merged_data['usage_percentage_array_away'][i] = np.array(list(temp_away.values()))
        merged_data['usage_percentage_array'][i] = np.concatenate((merged_data['usage_percentage_array_home'][i], merged_data['usage_percentage_array_away'][i]))
    return merged_data

def PER(merged_data, uniq_players):
    merged_data['PER_array_home'] = None
    merged_data['PER_array_away'] = None
    merged_data['PER_array'] = None
        
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
                            temp_home[x] = merged_data['players'][i][y]['player_efficiency_rating'] 
                        except:
                            temp_home[x] = 0
                    elif merged_data['players'][i][y]['team'] == merged_data['away_team'][i]:
                        try:
                            temp_away[x] = merged_data['players'][i][y]['player_efficiency_rating']
                        except:
                            temp_away[x] = 0
        merged_data['PER_array_home'][i] = np.array(list(temp_home.values()))
        merged_data['PER_array_away'][i] = np.array(list(temp_away.values()))
        merged_data['PER_array'][i] = np.concatenate((merged_data['PER_array_home'][i], merged_data['PER_array_away'][i]))
    return merged_data

def win_shares(merged_data, uniq_players):
    merged_data['win_shares_array_home'] = None
    merged_data['win_shares_array_away'] = None
    merged_data['win_shares_array'] = None
        
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
                            temp_home[x] = merged_data['players'][i][y]['win_shares'] 
                        except:
                            temp_home[x] = 0
                    elif merged_data['players'][i][y]['team'] == merged_data['away_team'][i]:
                        try:
                            temp_away[x] = merged_data['players'][i][y]['win_shares']
                        except:
                            temp_away[x] = 0
        merged_data['win_shares_array_home'][i] = np.array(list(temp_home.values()))
        merged_data['win_shares_array_away'][i] = np.array(list(temp_away.values()))
        merged_data['win_shares_array'][i] = np.concatenate((merged_data['win_shares_array_home'][i], merged_data['win_shares_array_away'][i]))
    return merged_data

def blocks(merged_data, uniq_players):
    merged_data['blocks_array_home'] = None
    merged_data['blocks_array_away'] = None
    merged_data['blocks_array'] = None
        
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
                            temp_home[x] = merged_data['players'][i][y]['block_percentage']
                        except:
                            temp_home[x] = 0
                    elif merged_data['players'][i][y]['team'] == merged_data['away_team'][i]:
                        try:
                            temp_away[x] = merged_data['players'][i][y]['block_percentage']
                        except:
                            temp_away[x] = 0
        merged_data['blocks_array_home'][i] = np.array(list(temp_home.values()))
        merged_data['blocks_array_away'][i] = np.array(list(temp_away.values()))
        merged_data['blocks_array'][i] = np.concatenate((merged_data['blocks_array_home'][i], merged_data['blocks_array_away'][i]))
    return merged_data



def assists(merged_data, uniq_players):
    merged_data['assists_array_home'] = None
    merged_data['assists_array_away'] = None
    merged_data['assists_array'] = None
        
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
                            temp_home[x] = merged_data['players'][i][y]['assist_percentage']
                        except:
                            temp_home[x] = 0
                    elif merged_data['players'][i][y]['team'] == merged_data['away_team'][i]:
                        try:
                            temp_away[x] = merged_data['players'][i][y]['assist_percentage']
                        except:
                            temp_away[x] = 0
        merged_data['assists_array_home'][i] = np.array(list(temp_home.values()))
        merged_data['assists_array_away'][i] = np.array(list(temp_away.values()))
        merged_data['assists_array'][i] = np.concatenate((merged_data['assists_array_home'][i], merged_data['assists_array_away'][i]))
    return merged_data

def minutes_played(merged_data, uniq_players):
    merged_data['minutes_played_array_home'] = None
    merged_data['minutes_played_array_away'] = None
    merged_data['minutes_played_array'] = None
        
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
                            temp_home[x] = merged_data['players'][i][y]['minutes_played']
                        except:
                            print("except")
                            temp_home[x] = 0
                    elif merged_data['players'][i][y]['team'] == merged_data['away_team'][i]:
                        try:
                            temp_away[x] = merged_data['players'][i][y]['minutes_played'] 
                        except:
                            print("except")
                            temp_away[x] = 0
        merged_data['minutes_played_array_home'][i] = np.array(list(temp_home.values()))
        merged_data['minutes_played_array_away'][i] = np.array(list(temp_away.values()))
        merged_data['minutes_played_array'][i] = np.concatenate((merged_data['minutes_played_array_home'][i], merged_data['minutes_played_array_away'][i]))
    return merged_data


def points(merged_data, uniq_players):
    merged_data['points_array_home'] = None
    merged_data['points_array_away'] = None
    merged_data['points_array'] = None
        
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
                            temp_home[x] = compute_points(merged_data['players'][i][y]['made_field_goals'], 
                                     merged_data['players'][i][y]['made_three_point_field_goals'],
                                     merged_data['players'][i][y]['made_free_throws'] ) / merged_data['players'][i][y]['games_played']
                        except:
                            temp_home[x] = 0
                    elif merged_data['players'][i][y]['team'] == merged_data['away_team'][i]:
                        try:
                            temp_away[x] = compute_points(merged_data['players'][i][y]['made_field_goals'], 
                                     merged_data['players'][i][y]['made_three_point_field_goals'],
                                     merged_data['players'][i][y]['made_free_throws'] ) / merged_data['players'][i][y]['games_played']
                        except:
                            temp_away[x] = 0
        merged_data['points_array_home'][i] = np.array(list(temp_home.values()))
        merged_data['points_array_away'][i] = np.array(list(temp_away.values()))
        merged_data['points_array'][i] = np.concatenate((merged_data['points_array_home'][i], merged_data['points_array_away'][i]))
    return merged_data


def efg(merged_data, uniq_players):
    merged_data['efg_array_home'] = None
    merged_data['efg_array_away'] = None
    merged_data['efg_array'] = None
        
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
                            temp_home[x] = compute_efg(merged_data['players'][i][y]['made_field_goals'], 
                                     merged_data['players'][i][y]['made_three_point_field_goals'],
                                     merged_data['players'][i][y]['attempted_field_goals']
                                      )
                        except:
                            print('landed in except')
                            temp_home[x] = 0
                    elif merged_data['players'][i][y]['team'] == merged_data['away_team'][i]:
                        try:
                            temp_away[x] = compute_efg(merged_data['players'][i][y]['made_field_goals'], 
                                     merged_data['players'][i][y]['made_three_point_field_goals'],
                                     merged_data['players'][i][y]['attempted_field_goals']
                                     )
                        except:
                            print('landed in except away')
                            temp_away[x] = 0
        merged_data['efg_array_home'][i] = np.array(list(temp_home.values()))
        merged_data['efg_array_away'][i] = np.array(list(temp_away.values()))
        merged_data['efg_array'][i] = np.concatenate((merged_data['efg_array_home'][i], merged_data['efg_array_away'][i]))
    return merged_data


def player_stats(merged_data, uniq_players):
    merged_data['stats_array_home'] = None
    merged_data['stats_array_away'] = None
    merged_data['stats_array'] = None
        
    for i in range(len(merged_data)):
        print(i)
        temp_home = {}
        temp_away = {}
        for x in uniq_players:
            temp_home[x] = [0,0,0,0]
            temp_away[x] = [0,0,0,0]
            for y in merged_data['players'][i].keys():
                if y == x:
                    if merged_data['players'][i][y]['team'] == merged_data['home_team'][i]:
                        try:
                            temp_home[x][0] = 1
                            temp_home[x][1] = compute_efg(merged_data['players'][i][y]['made_field_goals'], 
                                     merged_data['players'][i][y]['made_three_point_field_goals'],
                                     merged_data['players'][i][y]['attempted_field_goals'])
                            temp_home[x][2] = compute_points(merged_data['players'][i][y]['made_field_goals'], 
                                     merged_data['players'][i][y]['made_three_point_field_goals'],
                                     merged_data['players'][i][y]['made_free_throws'] ) / merged_data['players'][i][y]['games_played']
                            temp_home[x][3] = merged_data['players'][i][y]['minutes_played']
                        except:
                            temp_home[x] = [1,0,0,0]
                    elif merged_data['players'][i][y]['team'] == merged_data['away_team'][i]:
                        try:
                            temp_away[x][0] = 1
                            temp_away[x][1] = compute_efg(merged_data['players'][i][y]['made_field_goals'], 
                                     merged_data['players'][i][y]['made_three_point_field_goals'],
                                     merged_data['players'][i][y]['attempted_field_goals'])
                            temp_away[x][2] = compute_points(merged_data['players'][i][y]['made_field_goals'], 
                                     merged_data['players'][i][y]['made_three_point_field_goals'],
                                     merged_data['players'][i][y]['made_free_throws'] ) / merged_data['players'][i][y]['games_played']
                            temp_away[x][3] = merged_data['players'][i][y]['minutes_played']
                        except:
                            temp_away[x] = [1,0,0,0]
        merged_data['stats_array_home'][i] = np.array(list(temp_home.values()))
        merged_data['stats_array_away'][i] = np.array(list(temp_away.values()))
        merged_data['stats_array'][i] = np.concatenate((merged_data['stats_array_home'][i], merged_data['stats_array_away'][i]))
    return merged_data

def points_end(merged_data, uniq_players):
    merged_data['pointsend_array_home'] = None
    merged_data['pointsend_array_away'] = None
    merged_data['pointsend_array'] = None
        
    for i in range(len(merged_data)):
        print(i)
        temp_home = np.array([0,0])
        temp_away = np.array([0,0])
        for y in merged_data['players'][i].keys():
            if merged_data['players'][i][y]['team'] == merged_data['home_team'][i]:
                try:
                    temp_home[0] = compute_points(merged_data['players'][i][y]['made_field_goals'], 
                             merged_data['players'][i][y]['made_three_point_field_goals'],
                             merged_data['players'][i][y]['made_free_throws'] ) / merged_data['players'][i][y]['games_played']
                    temp_home[1] = compute_efg(merged_data['players'][i][y]['made_field_goals'], 
                                     merged_data['players'][i][y]['made_three_point_field_goals'],
                                     merged_data['players'][i][y]['attempted_field_goals'])
                except:
                    temp_home[x] = 0
            elif merged_data['players'][i][y]['team'] == merged_data['away_team'][i]:
                try:
                    temp_away[x] = compute_points(merged_data['players'][i][y]['made_field_goals'], 
                             merged_data['players'][i][y]['made_three_point_field_goals'],
                             merged_data['players'][i][y]['made_free_throws'] ) / merged_data['players'][i][y]['games_played']
                except:
                    temp_away[x] = 0
        merged_data['points_array_home'][i] = np.array(list(temp_home.values()))
        merged_data['points_array_away'][i] = np.array(list(temp_away.values()))
        merged_data['points_array'][i] = np.concatenate((merged_data['points_array_home'][i], merged_data['points_array_away'][i]) )
    return merged_data

def compute_points(FG, threeP, FT):
    threeP_pts = threeP * 3
    twoP_pts = (FG - threeP) * 2
    return threeP_pts + twoP_pts + FT
    
def compute_efg(FG, threeP, FGA):
    twoP_pts = FG - threeP
    return ((twoP_pts + 1.5 * threeP) / FGA)*100
    
  
    