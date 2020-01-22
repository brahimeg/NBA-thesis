# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:38:45 2020

@author: Ibrahim-main
"""
import pandas as pd 
import numpy as np
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType
import os
import datetime
from datetime import timedelta

#gather all game results sorted by date
def upload_data():
    client.season_schedule(season_end_year=2015, output_type=OutputType.CSV,  output_file_path = './game_results_2014_2015.csv')
    client.season_schedule(season_end_year=2016, output_type=OutputType.CSV,  output_file_path = './game_results_2015_2016.csv')
    client.season_schedule(season_end_year=2017, output_type=OutputType.CSV,  output_file_path = './game_results_2016_2017.csv')
    client.season_schedule(season_end_year=2018, output_type=OutputType.CSV,  output_file_path = './game_results_2017_2018.csv')

#gather and combine all data in one dataframe, including player-stats per game
def process_data():      
    all_data = pd.DataFrame()
    
    for file in os.listdir():
        temp = pd.read_csv(file)
        all_data = pd.concat([all_data, temp])    
        
    all_data.index = range(len(all_data))
    all_data = all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')]
    all_data['players'] = None 
    for i in range(25):
         all_data['start_time'][i] = datetime.datetime.strptime(all_data['start_time'][i][0:16], '%Y-%m-%d %H:%M')
         all_data['start_time'][i] = all_data['start_time'][i] - timedelta(hours = 4)
         all_data['players'][i] = []     
         for player in client.player_box_scores(day = all_data['start_time'][i].day, month = all_data['start_time'][i].month,
                                                year = all_data['start_time'][i].year):
             if player['team'].value == all_data['home_team'][i] or player['team'].value == all_data['away_team'][i]:
                 all_data['players'][i].append(player)
    
    return all_data



merged_data = process_data()
merged_data.to_csv('./merged_data.csv') 


test = client.player_box_scores(day=28, month=10, year=2014)






