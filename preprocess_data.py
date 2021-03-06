"""
Created on Tue Jan 21 17:38:45 2020

@author: Ibrahim-main
"""
import pandas as pd 
import numpy as np
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType
from basketball_reference_web_scraper.data import Team
import os
import glob
import datetime
from datetime import timedelta
import sklearn as sk
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neural_network import MLPClassifier


#gather all game results sorted by date
def upload_data():
    client.season_schedule(season_end_year=2014, output_type=OutputType.CSV,  output_file_path = './data/game_results_2013_2014.csv')
    client.season_schedule(season_end_year=2015, output_type=OutputType.CSV,  output_file_path = './data/game_results_2014_2015.csv')
    client.season_schedule(season_end_year=2016, output_type=OutputType.CSV,  output_file_path = './data/game_results_2015_2016.csv')
    client.season_schedule(season_end_year=2017, output_type=OutputType.CSV,  output_file_path = './data/game_results_2016_2017.csv')
    client.season_schedule(season_end_year=2018, output_type=OutputType.CSV,  output_file_path = './data/game_results_2017_2018.csv')
    
    client.players_season_totals(season_end_year=2013, output_type=OutputType.JSON,  output_file_path = './player_stats_2012_2013.json')
    client.players_season_totals(season_end_year=2014, output_type=OutputType.JSON,  output_file_path = './player_stats_2013_2014.json')
    client.players_season_totals(season_end_year=2015, output_type=OutputType.JSON,  output_file_path = './player_stats_2014_2015.json')
    client.players_season_totals(season_end_year=2016, output_type=OutputType.JSON,  output_file_path = './player_stats_2015_2016.json')    
    client.players_season_totals(season_end_year=2017, output_type=OutputType.JSON,  output_file_path = './player_stats_2016_2017.json')
    client.players_season_totals(season_end_year=2018, output_type=OutputType.JSON,  output_file_path = './player_stats_2017_2018.json')



#gather and combine all data in one dataframe, including player-stats per game
def process_gamedata():
    
    
    all_data = pd.DataFrame()
    player_stats = process_playerdata()
    
    counter = 0
    for file in glob.glob('./data/*.csv'):
        temp = pd.read_csv(file)
        temp['players'] = None
        for i in range(len(temp)):
            print(i)
            temp['start_time'][i] = datetime.datetime.strptime(temp['start_time'][i][0:16], '%Y-%m-%d %H:%M')
            temp['start_time'][i] = temp['start_time'][i] - timedelta(hours = 4)
            temp['players'][i] = {}
            if (counter == 0):
                for x in player_stats['2014'].keys():
                    if temp['home_team'][i] == player_stats['2014'][x]['team'] or temp['away_team'][i] == player_stats['2014'][x]['team']:
                        try:
                            temp['players'][i][x] = player_stats['2013'][x]
                            temp['players'][i][x]['team'] = player_stats['2014'][x]['team']
                        except KeyError:
                            temp['players'][i][x] = pd.Series({'team': player_stats['2014'][x]['team']})
            elif (counter == 1):
                for x in player_stats['2015'].keys():
                    if temp['home_team'][i] == player_stats['2015'][x]['team'] or temp['away_team'][i] == player_stats['2015'][x]['team']:
                        try:
                            temp['players'][i][x] = player_stats['2014'][x]
                            temp['players'][i][x]['team'] = player_stats['2015'][x]['team']
                        except KeyError:
                            temp['players'][i][x] = pd.Series({'team': player_stats['2015'][x]['team']})                
            elif (counter == 2):
                for x in player_stats['2016'].keys():
                    if temp['home_team'][i] == player_stats['2016'][x]['team'] or temp['away_team'][i] == player_stats['2016'][x]['team']:
                        try:
                            temp['players'][i][x] = player_stats['2015'][x]
                            temp['players'][i][x]['team'] = player_stats['2016'][x]['team']
                        except KeyError:
                            temp['players'][i][x] = pd.Series({'team' : player_stats['2016'][x]['team']})
            elif (counter == 3):
                for x in player_stats['2017'].keys():
                    if temp['home_team'][i] == player_stats['2017'][x]['team'] or temp['away_team'][i] == player_stats['2017'][x]['team']:
                        try:
                            temp['players'][i][x] = player_stats['2016'][x]
                            temp['players'][i][x]['team'] = player_stats['2017'][x]['team']
                        except KeyError:
                            temp['players'][i][x] = pd.Series({'team' : player_stats['2017'][x]['team']})
            elif (counter == 4):
                for x in player_stats['2018'].keys():
                    if temp['home_team'][i] == player_stats['2018'][x]['team'] or temp['away_team'][i] == player_stats['2018'][x]['team']:
                        try:
                            temp['players'][i][x] = player_stats['2017'][x]
                            temp['players'][i][x]['team'] = player_stats['2018'][x]['team']
                        except KeyError:
                            temp['players'][i][x] = pd.Series({'team': player_stats['2018'][x]['team']})
        counter += 1
        all_data = pd.concat([all_data, temp])
    
    all_data.index = range(len(all_data))
    all_data = all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')]
    
    all_data['home_players'] = None
    all_data['away_players'] =  None
    for i in range(len(all_data)):
        print('i',i)
        all_data['home_players'][i] = {}
        all_data['away_players'][i] = {}
        for player in all_data['players'][i].keys():
            if all_data['players'][i][player]['team'] == all_data['home_team'][i]:
                all_data['home_players'][i][player] = all_data['players'][i][player]
            else:
                all_data['away_players'][i][player] = all_data['players'][i][player]
    
    
    return all_data



#gather all seasonal player-stats and store them by player
def process_playerdata():
    stats_2013 = pd.read_json('./data/player_stats_2012_2013.json')
    stats_2014 = pd.read_json('./data/player_stats_2013_2014.json')
    stats_2015 = pd.read_json('./data/player_stats_2014_2015.json')
    stats_2016 = pd.read_json('./data/player_stats_2015_2016.json')
    stats_2017 = pd.read_json('./data/player_stats_2016_2017.json')
    stats_2018 = pd.read_json('./data/player_stats_2017_2018.json')
   
    player_stats = {}
    player_stats['2013'] = {}
    player_stats['2014'] = {}
    player_stats['2015'] = {}
    player_stats['2016'] = {}
    player_stats['2017'] = {}
    player_stats['2018'] = {}
    
    for x in range(len(stats_2013)):
        if stats_2013['name'][x] not in player_stats['2013'].keys():           
            player_stats['2013'][stats_2013['name'][x]] = stats_2013.loc[x]
        else:
            if player_stats['2013'][stats_2013['name'][x]]['games_played'] > stats_2013.loc[x]['games_played']:
                continue
            else:
                player_stats['2013'][stats_2013['name'][x]] = stats_2013.loc[x]
    for x in range(len(stats_2014)):
        if stats_2014['name'][x] not in player_stats['2014'].keys():           
            player_stats['2014'][stats_2014['name'][x]] = stats_2014.loc[x]
        else:
            if player_stats['2014'][stats_2014['name'][x]]['games_played'] > stats_2014.loc[x]['games_played']:
                continue
            else:
                player_stats['2014'][stats_2014['name'][x]] = stats_2014.loc[x]
    for x in range(len(stats_2015)):
        if stats_2015['name'][x] not in player_stats['2015'].keys():           
            player_stats['2015'][stats_2015['name'][x]] = stats_2015.loc[x]
        else:
            if player_stats['2015'][stats_2015['name'][x]]['games_played'] > stats_2015.loc[x]['games_played']:
                continue
            else:
                player_stats['2015'][stats_2015['name'][x]] = stats_2015.loc[x]
    for x in range(len(stats_2016)):
        if stats_2016['name'][x] not in player_stats['2016'].keys():           
            player_stats['2016'][stats_2016['name'][x]] = stats_2016.loc[x]
        else:
            if player_stats['2016'][stats_2016['name'][x]]['games_played'] > stats_2016.loc[x]['games_played']:
                continue
            else:
                player_stats['2016'][stats_2016['name'][x]] = stats_2016.loc[x]          
    for x in range(len(stats_2017)):
        if stats_2017['name'][x] not in player_stats['2017'].keys():           
            player_stats['2017'][stats_2017['name'][x]] = stats_2017.loc[x]
        else:
            if player_stats['2017'][stats_2017['name'][x]]['games_played'] > stats_2017.loc[x]['games_played']:
                continue
            else:
                player_stats['2017'][stats_2017['name'][x]] = stats_2017.loc[x]
    for x in range(len(stats_2018)):
        if stats_2018['name'][x] not in player_stats['2018'].keys():           
            player_stats['2018'][stats_2018['name'][x]] = stats_2018.loc[x]
        else:
            if player_stats['2018'][stats_2018['name'][x]]['games_played'] > stats_2018.loc[x]['games_played']:
                continue
            else:
                player_stats['2018'][stats_2018['name'][x]] = stats_2018.loc[x]
   
    return player_stats

#def unique_playerstats():
#    player_data = process_playerdata()
#    teams = unique_teams()
#    for year in player_data:
#        for player in year:
        

def unique_players():
    player_data = process_playerdata()
    uniq_players = {None}
    set(uniq_players)
    years = ['2014','2015', '2016', '2017', '2018']
    for i in years:
        for x in player_data[i].keys():
            uniq_players.add(x)
    
    return uniq_players

def unique_teams(merged_data):
    uniq_teams = {None}
    set(uniq_teams)
    for i in range(len(merged_data)):
        uniq_teams.add(merged_data['home_team'][i])
        
    uniq_teams.remove(None)
    return uniq_teams


def create_playerarray():
    merged_data, uniq_players = process_gamedata(), unique_players()
    
    #creating identity arrays for teams for every game
    merged_data['player_array_home'] = None
    
    merged_data['player_array_away'] = None
    merged_data['player_array'] = None
    merged_data['win'] = None
    
    for i in range(len(merged_data)):
        print("2nd", i)
        if merged_data['home_team_score'][i] > merged_data['away_team_score'][i]:
            merged_data['win'][i] = 1           
        else:
            merged_data['win'][i] = 0
        temp_home = {}
        temp_away = {}  
        for x in uniq_players:
            temp_home[x] = 0
            temp_away[x] = 0
            for y in merged_data['players'][i].keys():
                if y == x:
                    if merged_data['players'][i][y]['team'] == merged_data['home_team'][i]:a
                        temp_home[x] = 1
                    elif merged_data['players'][i][y]['team'] == merged_data['away_team'][i]:
                        temp_away[x] = 1 
                        
        merged_data['player_array_home'][i] = np.array(list(temp_home.values()))
        merged_data['player_array_away'][i] = np.array(list(temp_away.values()))
        merged_data['player_array'][i] = np.concatenate((merged_data['player_array_home'][i], merged_data['player_array_away'][i]))
    return merged_data



if __name__ == "__main__":
    #create_playerarray().to_json('./data/merged_data.json')
    print('test')

