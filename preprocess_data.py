"""
Created on Tue Jan 21 17:38:45 2020

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

#gather all game results sorted by date
def upload_data():
    client.season_schedule(season_end_year=2015, output_type=OutputType.CSV,  output_file_path = './game_results_2014_2015.csv')
    client.season_schedule(season_end_year=2016, output_type=OutputType.CSV,  output_file_path = './game_results_2015_2016.csv')
    client.season_schedule(season_end_year=2017, output_type=OutputType.CSV,  output_file_path = './game_results_2016_2017.csv')
    client.season_schedule(season_end_year=2018, output_type=OutputType.CSV,  output_file_path = './game_results_2017_2018.csv')
    
    client.players_season_totals(season_end_year=2015, output_type=OutputType.JSON,  output_file_path = './player_stats_2014_2015.json')
    client.players_season_totals(season_end_year=2016, output_type=OutputType.JSON,  output_file_path = './player_stats_2015_2016.json')    
    client.players_season_totals(season_end_year=2017, output_type=OutputType.JSON,  output_file_path = './player_stats_2016_2017.json')
    client.players_season_totals(season_end_year=2018, output_type=OutputType.JSON,  output_file_path = './player_stats_2017_2018.json')



#gather and combine all data in one dataframe, including player-stats per game
def process_gamedata():
    

##Adds individual game stats to all pleyer per game
#    for i in range(len(all_data)):
#        print(i)   
#        all_data['start_time'][i] = datetime.datetime.strptime(all_data['start_time'][i][0:16], '%Y-%m-%d %H:%M')
#        all_data['start_time'][i] = all_data['start_time'][i] - timedelta(hours = 4)
#        all_data['players'][i] = []
#        for player in client.player_box_scores(day = all_data['start_time'][i].day, month = all_data['start_time'][i].month,
#                                               year = all_data['start_time'][i].year):
#            if player['team'].value == all_data['home_team'][i] or player['team'].value == all_data['away_team'][i]:
#                all_data['players'][i].append(player)
    
    all_data = pd.DataFrame()
    player_stats = process_playerdata()
    
    counter = 0
    for file in glob.glob('*.csv'):
        temp = pd.read_csv(file)
        temp['players'] = None
        for i in range(len(temp)):
            print(i)
            temp['start_time'][i] = datetime.datetime.strptime(temp['start_time'][i][0:16], '%Y-%m-%d %H:%M')
            temp['start_time'][i] = temp['start_time'][i] - timedelta(hours = 4)
            date = temp['start_time'][i].year
            temp['players'][i] = {}
            if (counter == 0):
                for x in player_stats['2015'].keys():
                    if temp['home_team'][i] == player_stats['2015'][x]['team'] or temp['away_team'][i] == player_stats['2015'][x]['team']:
                        temp['players'][i][x] = player_stats['2015'][x]
            elif (counter == 1):
                for x in player_stats['2016'].keys():
                    if temp['home_team'][i] == player_stats['2016'][x]['team'] or temp['away_team'][i] == player_stats['2016'][x]['team']:
                        temp['players'][i][x] = player_stats['2016'][x]
            elif (counter == 2):
                for x in player_stats['2017'].keys():
                    if temp['home_team'][i] == player_stats['2017'][x]['team'] or temp['away_team'][i] == player_stats['2017'][x]['team']:
                        temp['players'][i][x] = player_stats['2017'][x]
            elif (counter == 3):
                for x in player_stats['2018'].keys():
                    if temp['home_team'][i] == player_stats['2018'][x]['team'] or temp['away_team'][i] == player_stats['2018'][x]['team']:
                        temp['players'][i][x] = player_stats['2018'][x]
        counter += 1
        all_data = pd.concat([all_data, temp])
    
    all_data.index = range(len(all_data))
    all_data = all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')]
    
    return all_data



merged_data = process_gamedata()
#merged_data.to_json('./merged_data.json') 



#gather all seasonal player-stats and store them by player
def process_playerdata():
    stats_2015 = pd.read_json('player_stats_2014_2015.json')
    stats_2016 = pd.read_json('player_stats_2015_2016.json')
    stats_2017 = pd.read_json('player_stats_2016_2017.json')
    stats_2018 = pd.read_json('player_stats_2017_2018.json')
    
    player_stats = {}
    player_stats['2015'] = {}
    player_stats['2016'] = {}
    player_stats['2017'] = {}
    player_stats['2018'] = {}
    
    for x in range(len(stats_2015)):
        player_stats['2015'][stats_2015['name'][x]] = stats_2015.loc[x]
    for x in range(len(stats_2016)):
        player_stats['2016'][stats_2016['name'][x]] = stats_2016.loc[x]           
    for x in range(len(stats_2017)):
        player_stats['2017'][stats_2017['name'][x]] = stats_2017.loc[x]
    for x in range(len(stats_2015)):
        player_stats['2018'][stats_2018['name'][x]] = stats_2018.loc[x]
    
    
    
    return player_stats









if __name__ == "__main__":
    print('class')