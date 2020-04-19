# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:57:44 2020

@author: Ibrahim-main
"""


def calc_avg(merged_data):
    stat_attr = ['age', 'assists', 'attempted_field_goals', 'attempted_free_throws', 'attempted_three_point_field_goals', 
     'blocks', 'defensive_rebounds', 'games_played', 'games_started', 'made_field_goals', 'made_free_throws', 
     'made_three_point_field_goals', 'minutes_played', 'name', 
     'offensive_rebounds', 'personal_fouls', 'positions', 'slug', 'steals', 'team', 'turnovers']
    
    merged_data['home_averages'] = None
    merged_data['away_averages'] = None
    problem = False
    
    for i in range(len(merged_data)):
        print(i)
        temp_home = {}
        temp_away = {}
        for attr in stat_attr:
            home_aver = []
            away_aver = []
            for player in merged_data['players'][i].keys():
                if merged_data['home_team'][i] == merged_data['players'][i][player]['team']:               
                    try:  
                        if type(merged_data['home_players'][i][player][attr]) == int:
                            home_aver.append(merged_data['home_players'][i][player][attr])
                        else:
                            problem = True
                    except KeyError:
                        continue
                if merged_data['away_team'][i] == merged_data['players'][i][player]['team']:               
                    try:  
                        if type(merged_data['away_players'][i][player][attr]) == int:
                            away_aver.append(merged_data['away_players'][i][player][attr])
                        else:
                            problem = True
                    except KeyError:
                        continue        
            
            if problem:
                problem = False
                continue
            
            temp_home[attr] = round(sum(home_aver) / len(home_aver),2)           
            temp_away[attr] = round(sum(away_aver) / len(away_aver),2)
            
        merged_data['home_averages'][i] = temp_home
        merged_data['away_averages'][i] = temp_away
    
    return merged_data


def fill_missing(merged_data):
    stripped_stat_attr = ['age', 'assists', 'attempted_field_goals', 'attempted_free_throws', 'attempted_three_point_field_goals', 
     'blocks', 'defensive_rebounds', 'games_played', 'games_started', 'made_field_goals', 'made_free_throws', 
     'made_three_point_field_goals', 'minutes_played', 
     'offensive_rebounds', 'personal_fouls', 'steals', 'turnovers']    
                  
    for i in range(len(merged_data)):
        print(i)
        for attr in stripped_stat_attr:
            for player in merged_data['players'][i].keys():
                if merged_data['home_team'][i] == merged_data['players'][i][player]['team']: 
                    try:
                        merged_data['home_players'][i][player][attr] = merged_data['home_players'][i][player][attr]
                    except:
                        merged_data['players'][i][player][attr] = merged_data['home_averages'][i][attr]
                        merged_data['home_players'][i][player][attr] = merged_data['home_averages'][i][attr]
                        
                if merged_data['away_team'][i] == merged_data['players'][i][player]['team']:
                    try:
                        merged_data['away_players'][i][player][attr] = merged_data['away_players'][i][player][attr]
                    except:
                        merged_data['players'][i][player][attr] = merged_data['away_averages'][i][attr]  
                        merged_data['away_players'][i][player][attr] = merged_data['away_averages'][i][attr]  
    
    return merged_data









