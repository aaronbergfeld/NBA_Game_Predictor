import torch
from torch.utils.data import Dataset
import numpy as np
from utils import TEAM_NAME_TO_ID, standardize_data
import pandas as pd

class GamesDataset(Dataset):
    def __init__(self, game_data, player_data, team_data, odds_data=None):
        self.games = game_data
        self.players = player_data
        self.team_data = team_data
        # self.players = self.players.groupby(['GAME_ID', 'TEAM_ID']).filter(lambda x: len(x) >= 10)
        self.odds = odds_data
        self.game_ids = self.games['GAME_ID'].unique()
        
        
    def __len__(self):
        return len(self.game_ids.tolist())

    def __getitem__(self, idx):
        game_id = self.game_ids[idx]
        game = self.games[self.games['GAME_ID'] == game_id]
        home_team = game[game['HOME_TEAM'] == 1]
        away_team = game[game['HOME_TEAM'] == 0]
        home_team_players = self.players[(self.players['GAME_ID'] == game_id) & (self.players['TEAM_ID'] == home_team['TEAM_ID'].values[0])]
        away_team_players = self.players[(self.players['GAME_ID'] == game_id) & (self.players['TEAM_ID'] == away_team['TEAM_ID'].values[0])]
        # Sort players by seconds played
        home_team_players = home_team_players.sort_values(by='SEC', ascending=False)
        away_team_players = away_team_players.sort_values(by='SEC', ascending=False)
        home_team_players = home_team_players.iloc[:10]
        away_team_players = away_team_players.iloc[:10]
        # Drop columns that are not needed
        home_team_players.drop(columns=['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'GAME_DATE', 'SEASON'], inplace=True)
        away_team_players.drop(columns=['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'GAME_DATE', 'SEASON'], inplace=True)
        # Convert to numpy array
        home_team_players = home_team_players.to_numpy()
        away_team_players = away_team_players.to_numpy()
        input = np.array([home_team_players, away_team_players])
        # Convert to tensor
        input = torch.from_numpy(input).float()
        # Get outcome
        label = game['HOME_TEAM_WON'].values[0]
        # Get team data
        home_team_data = self.team_data[self.team_data['TEAM_ID'] == home_team['TEAM_ID'].values[0]].sort_values(by='GAME_ID')
        away_team_data = self.team_data[self.team_data['TEAM_ID'] == away_team['TEAM_ID'].values[0]].sort_values(by='GAME_ID')
        home_team_data = home_team_data.iloc[-1]
        away_team_data = away_team_data.iloc[-1]
        home_team_data = home_team_data.drop(['TEAM_ID', 'GAME_ID'])
        away_team_data = away_team_data.drop(['TEAM_ID', 'GAME_ID'])
        home_team_data = home_team_data.to_numpy()
        away_team_data = away_team_data.to_numpy()
        team_data = np.array([home_team_data, away_team_data])
        team_data = torch.from_numpy(team_data).float()
        # Get odds
        if self.odds is None:
            return input, team_data, label
        odds = self.odds[self.odds['GAME_ID'] == game['GAME_ID'].values[0]]
        home_odds = odds['best_home_odds'].values[0]
        away_odds = odds['best_away_odds'].values[0]
        return input, team_data, label, home_odds, away_odds

class UpcomingGamesDataset(Dataset):
    def __init__(self, upcoming_games, num_players, player_data, team_data, model_version, injury_report=None):
        self.games = upcoming_games
        self.num_players = num_players
        self.players = player_data
        self.team_data = team_data
        self.injury_report = injury_report
        self.model_version = model_version

    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, idx):
        game = self.games.iloc[idx]
        home_team = game['home_team']
        away_team = game['away_team']
        home_team_id = TEAM_NAME_TO_ID[home_team]
        away_team_id = TEAM_NAME_TO_ID[away_team]
        # Get players
        home_team_players = self.players[(self.players['TEAM_ID'] == home_team_id) & (self.players['SEASON'] == '2023-2024')] 
        away_team_players = self.players[(self.players['TEAM_ID'] == away_team_id) & (self.players['SEASON'] == '2023-2024')]
        # Get last game for each player
        home_team_players = home_team_players.sort_values(by='GAME_DATE', ascending=False)
        away_team_players = away_team_players.sort_values(by='GAME_DATE', ascending=False)
        home_team_players = home_team_players.groupby('PLAYER_ID').first().reset_index()
        away_team_players = away_team_players.groupby('PLAYER_ID').first().reset_index()
        # Remove players whos status is 'Out'
        if self.injury_report is not None:
            players_out = self.injury_report[self.injury_report['STATUS'] == 'Out']
            home_team_players = home_team_players[~home_team_players['PLAYER_ID'].isin(players_out['PLAYER_ID'])]
            away_team_players = away_team_players[~away_team_players['PLAYER_ID'].isin(players_out['PLAYER_ID'])]
        # Update 'DAYS_SINCE_LAST_GAME' column
        home_team_players.loc[:, 'DAYS_SINCE_LAST_GAME'] = (pd.to_datetime('today') - pd.to_datetime(home_team_players['GAME_DATE'])).dt.days
        away_team_players.loc[:, 'DAYS_SINCE_LAST_GAME'] = (pd.to_datetime('today') - pd.to_datetime(away_team_players['GAME_DATE'])).dt.days

        home_team_players = home_team_players.sort_values(by='SEC', ascending=False)
        away_team_players = away_team_players.sort_values(by='SEC', ascending=False)
        print(idx, home_team_players.shape, away_team_players.shape)
        if home_team_players.shape[0] < self.num_players:
            zero_rows = pd.DataFrame(0, index=np.arange(self.num_players-home_team_players.shape[0]), columns=home_team_players.columns)
            home_team_players = pd.concat([home_team_players, zero_rows])
        if away_team_players.shape[0] < self.num_players:
            zero_rows = pd.DataFrame(
                0,
                index=np.arange(self.num_players - away_team_players.shape[0]),
                columns=away_team_players.columns,
            )
            away_team_players = pd.concat([away_team_players, zero_rows])
        print(idx, home_team_players.shape, away_team_players.shape)
        home_team_players = home_team_players.iloc[:self.num_players]
        away_team_players = away_team_players.iloc[:self.num_players]
        print(idx, home_team_players.shape, away_team_players.shape)
        # Drop columns that are not needed
        home_team_players.drop(columns=['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'GAME_DATE', 'SEASON'], inplace=True)
        away_team_players.drop(columns=['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'GAME_DATE', 'SEASON'], inplace=True)
        # Get team data
        home_team_data = self.team_data[self.team_data['TEAM_ID'] == home_team_id].sort_values(by='GAME_ID')
        away_team_data = self.team_data[self.team_data['TEAM_ID'] == away_team_id].sort_values(by='GAME_ID')
        home_team_data = home_team_data.iloc[-1]
        away_team_data = away_team_data.iloc[-1]
        home_team_data = home_team_data.drop(['TEAM_ID', 'GAME_ID'])
        away_team_data = away_team_data.drop(['TEAM_ID', 'GAME_ID'])
        # Standardize data
        home_team_players, home_team_data = standardize_data(self.model_version, home_team_players, home_team_data) 
        away_team_players, away_team_data = standardize_data(self.model_version, away_team_players, away_team_data)
        # Convert to numpy array
        home_team_players = home_team_players.to_numpy()
        away_team_players = away_team_players.to_numpy()
        input = np.array([home_team_players, away_team_players])
        # Convert to tensor
        input = torch.from_numpy(input).float()
        home_team_data = home_team_data.to_numpy()
        away_team_data = away_team_data.to_numpy()
        team_data = np.array([home_team_data, away_team_data])
        
        return input, team_data, home_team, away_team 
