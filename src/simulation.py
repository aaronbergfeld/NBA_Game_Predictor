import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import importlib
import dataloaders
importlib.reload(dataloaders)
from dataloaders import GamesDataset
import torch

# bankroll = 1000
# # Define bettring rules to test
# hundreth_uniform = uniform_betting(bankroll, bankroll_fraction=0.01)
# tenths_uniform = uniform_betting(bankroll, bankroll_fraction=0.1)
# tenths_kelly = kelly_criterion(kelly_fraction=0.1)
# eighth_kelly = kelly_criterion(kelly_fraction=0.125)
# sixth_kelly = kelly_criterion(kelly_fraction=0.166)
# tenths_uniform_threshold_1 = uniform_betting(bankroll, bankroll_fraction=0.1, threshold=0.1)
# tenths_uniform_threshold_2 = uniform_betting(bankroll, bankroll_fraction=0.1, threshold=0.2)
# tenths_uniform_threshold_3 = uniform_betting(bankroll, bankroll_fraction=0.1, threshold=0.3)
# eighth_kelly_threshold_1 = kelly_criterion(kelly_fraction=0.125, threshold=0.1)
# eighth_kelly_threshold_2 = kelly_criterion(kelly_fraction=0.125, threshold=0.2)
# eighth_kelly_threshold_3 = kelly_criterion(kelly_fraction=0.125, threshold=0.3)

# betting_rules = []
# betting_rules.append(hundreth_uniform)
# betting_rules.append(tenths_uniform)
# betting_rules.append(tenths_kelly)
# betting_rules.append(eighth_kelly)
# betting_rules.append(sixth_kelly)
# betting_rules.append(tenths_uniform_threshold_1)
# betting_rules.append(tenths_uniform_threshold_2)
# betting_rules.append(tenths_uniform_threshold_3)
# betting_rules.append(eighth_kelly_threshold_1)
# betting_rules.append(eighth_kelly_threshold_2)
# betting_rules.append(eighth_kelly_threshold_3)

# Define betting rules to test
def uniform_betting(initial_bankroll, bankroll_fraction=0.1, threshold=0):
    def rule(bankroll, odds, win_probability):
        if abs(win_probability - .5) < threshold:
            return 0
        return initial_bankroll * bankroll_fraction
    return rule

def kelly_criterion(kelly_fraction=0.1, threshold=0):
    def rule(bankroll, odds, win_probability): 
        if abs(win_probability - .5) < threshold:
            return 0
        return bankroll * kelly_fraction * (odds * win_probability - (1 - win_probability))/odds
    return rule

def run_simulation(bankroll, betting_rules, model, season):
    """Simulate betting on games using a model and betting rules

    Args:
        bankroll (int): Initial bankroll
        betting_rules (list): List of betting rules to test
        model (torch.nn.Module): Model to use for predictions
        start_season (int, optional): First season to bet on. Defaults to 22020.
        end_season (int, optional): Last season to bet on. Defaults to 22023.

    Returns:
        bankroll_history (np.array): Bankroll history
        odds_history (np.array): Odds history
        bet_history (np.array): Bet history
        win_history (np.array): Win history
        model_predictions (np.array): Model predictions
        bookmaker_predictions (np.array): Bookmaker predictions
        labels (np.array): Labels
    """
    
    player_data = pd.read_csv('../data/rolling_averages.csv')
    game_data = pd.read_csv('../data/games.csv')
    odds_data = pd.read_csv('../data/odds.csv')
    team_level_data = pd.read_csv('../data/team_level_data.csv')

    # Drop games that does not have data for both teams
    player_data = player_data.groupby(["GAME_ID", "TEAM_ID"]).filter(lambda x: len(x) >= 10)
    player_data = player_data.groupby('GAME_ID').filter(lambda x: len(x['TEAM_ID'].unique()) == 2)

    SEASON_STRING_TO_ID = {
        '2020-2021': 22020,
        '2021-2022': 22021,
        '2022-2023': 22022,
        '2023-2024': 22023,
    }

    # only keep games in the 2022-23 regular season
    game_data['GAME_DATE'] = pd.to_datetime(game_data['GAME_DATE'])
    game_data = game_data[game_data['SEASON_ID'] == SEASON_STRING_TO_ID[season]]
    game_data.sort_values(by='GAME_DATE', inplace=True)

    # only keep games with odds
    game_data = game_data[game_data['GAME_ID'].isin(odds_data['GAME_ID'])]

    # only keep games with rolling averages
    game_data = game_data[game_data['GAME_ID'].isin(player_data['GAME_ID'])]

    dataset = GamesDataset(game_data, player_data, team_level_data, odds_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    bankroll_history = np.zeros((len(dataset) + 1, len(betting_rules)))
    # Set the first row to the initial bankroll
    bankroll_history[0, :] = bankroll

    odds_history = np.zeros(len(dataset))
    bet_history = np.zeros((len(dataset), len(betting_rules)))
    win_history = np.zeros(len(dataset))

    sigmoid = torch.nn.Sigmoid()

    # Load model
    model.eval()

    model_predictions = []
    bookmaker_predictions = []
    labels = []
    bets_placed = 0
    bets_won = 0
    for i, (input, team_data, label, home_odds, away_odds) in enumerate(dataloader):
        input = input.to(dtype=torch.float32, device='cuda')
        team_data = team_data.to(dtype=torch.float32, device='cuda')
        label = label.item()
        output = sigmoid(model(input, team_data))
        output = output.item()
        model_predictions.append(output)
        home_odds = home_odds.item()
        away_odds = away_odds.item()
        labels.append(label)
        if 1/home_odds > 1/away_odds:
            bookmaker_predictions.append(1)
        else:
            bookmaker_predictions.append(0)
        home_prediction = output
        away_prediction = 1 - output
        # choose the highest positive ev bet
        bet_on_home = home_prediction - 1/home_odds > away_prediction - 1/away_odds 
        
        if bet_on_home and home_prediction > 1/home_odds:
            for j, item in enumerate(betting_rules.items()):
                _, rule = item
                current_bankroll = bankroll_history[bets_placed, j]
                bet = rule(current_bankroll, home_odds, home_prediction) if current_bankroll > 0 else 0
                odds_history[bets_placed] = home_odds
                bet_history[bets_placed, j] = bet
                new_bankroll = current_bankroll - bet
                if label == 1:
                    new_bankroll += bet * home_odds
                bankroll_history[bets_placed+1, j] = new_bankroll
            if label == 1:
                bets_won += 1
                win_history[bets_placed] = 1
            else:
                win_history[bets_placed] = 0
            bets_placed += 1
        elif not bet_on_home and away_prediction > 1/away_odds:
            for j, item in enumerate(betting_rules.items()):
                _, rule = item
                current_bankroll = bankroll_history[bets_placed, j]
                bet = rule(current_bankroll, home_odds, home_prediction) if current_bankroll > 0 else 0
                odds_history[bets_placed] = away_odds
                bet_history[bets_placed, j] = bet
                new_bankroll = current_bankroll - bet
                if label == 0:
                    new_bankroll += bet * away_odds
                bankroll_history[bets_placed+1, j] = new_bankroll 
            if label == 0:
                bets_won += 1
                win_history[bets_placed] = 1
            else:
                win_history[bets_placed] = 0
            bets_placed += 1

    return bankroll_history, odds_history, bet_history, win_history, model_predictions, bookmaker_predictions, labels, bets_placed

def plot_bankroll_histories(bankroll_history, bets_placed, betting_rules):
    # Trim unused rows from bankroll history
    trimmed_bankroll_history = bankroll_history[:bets_placed+1, :]

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Plot each bankroll history
    for i, item in enumerate(betting_rules.items()):
        title, _ = item
        plt.plot(trimmed_bankroll_history[:, i], label=title)

    # Add a legend and show the plot
    plt.legend()
    plt.show()

def plot_bankroll_history(bankroll_history, bets_placed, betting_rule_title, betting_rules):
    # Trim unused rows from bankroll history
    betting_rule_index = [i for i, item in enumerate(betting_rules.items()) if item[0] == betting_rule_title][0]
    trimmed_bankroll_history = bankroll_history[:bets_placed+1, betting_rule_index]

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Plot each bankroll history
    plt.plot(trimmed_bankroll_history, label=betting_rule_title)

    # Add a legend and show the plot
    plt.legend()
    plt.show()

def print_statistics(model_predictions, labels, bookmaker_predictions, bets_placed, bets_won, bankroll_history):
    model_accuracy = np.sum(np.round(np.array(model_predictions)) == np.array(labels)) / len(labels)
    bookmaker_accuracy = np.sum(np.array(bookmaker_predictions) == np.array(labels)) / len(labels)
    pct_bets_placed = bets_placed / len(labels)
    pct_bets_won = bets_won / bets_placed
    best_performing_rule = np.argmax(bankroll_history[-1, :])

    print('Model accuracy: ', model_accuracy)
    print('Bookmaker accuracy: ', bookmaker_accuracy)
    print('PCT bets placed: ', pct_bets_placed)
    print('PCT bets won: ', pct_bets_won)
    print('Best performing rule: ', best_performing_rule, ' with bankroll: ', bankroll_history[-1, best_performing_rule])
