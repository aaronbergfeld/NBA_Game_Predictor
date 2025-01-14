# %%
import nba_api.stats.endpoints as nba
import pandas as pd
import numpy as np
import tqdm
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import date
from model_definitions import *

SEASONS = ['2005-06', '2006-07', '2007-08', '2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']

# %%
# Get all games since 2010-11 season
games = pd.DataFrame()
for season in SEASONS:
    print('Getting games for season: {}'.format(season))
    games = pd.concat([games, nba.leaguegamelog.LeagueGameLog(season=season).get_data_frames()[0]])

# %%
print('Total games: {}'.format(len(games)))

# Add a column indicating whether the team was home or away
games['HOME_TEAM'] = np.where(games['MATCHUP'].str.contains('vs.'), 1, 0)

# Add a column indicating whether the home team won based on the WL and HOME_TEAM columns
games['HOME_TEAM_WON'] = np.where((games['WL'] == 'W') & (games['HOME_TEAM'] == 1) | (games['WL'] == 'L') & (games['HOME_TEAM'] == 0), 1, 0)

# Save games to csv
games.to_csv('../data/games.csv', index=False)

games = pd.read_csv('../data/games.csv')

# Calculate the 10 game win percentage for each team
games = pd.read_csv('../data/games.csv')
team_win_percentage = pd.DataFrame()

for season in games['SEASON_ID'].unique():
    season_games = games[games['SEASON_ID'] == season]
    for team in season_games['TEAM_ID'].unique():
        team_games = season_games[season_games['TEAM_ID'] == team].sort_values('GAME_DATE')
        ten_game_win_percentage = team_games['WL'].apply(lambda x: 1 if x == 'W' else 0).rolling(window=10, min_periods=10).mean()
        cumilative_win_percentage = team_games['WL'].apply(lambda x: 1 if x == 'W' else 0).expanding().mean()
        home_win_percentage = team_games[team_games['HOME_TEAM'] == 1]['WL'].apply(lambda x: 1 if x == 'W' else 0).expanding().mean()
        away_win_percentage = team_games[team_games['HOME_TEAM'] == 0]['WL'].apply(lambda x: 1 if x == 'W' else 0).expanding().mean()
        win_percentages = pd.DataFrame({
            'GAME_ID': team_games['GAME_ID'],
            'TEAM_ID': team_games['TEAM_ID'],
            '10_GAME_WIN_PCT': ten_game_win_percentage,
            'CUM_WIN_PCT': cumilative_win_percentage,
            'HOME_WIN_PCT': home_win_percentage,
            'AWAY_WIN_PCT': away_win_percentage
        })
        win_percentages['GAME_ID'] = win_percentages['GAME_ID'].shift(-1)
        win_percentages.fillna(method='ffill', inplace=True)
        win_percentages.dropna(inplace=True)
        team_win_percentage = pd.concat([team_win_percentage, win_percentages], ignore_index=True)

print(len(team_win_percentage))

# Remove games that do not have 2 teams
team_win_percentage = team_win_percentage.groupby('GAME_ID').filter(lambda x: len(x) == 2)
team_win_percentage.dropna(inplace=True)

print(len(team_win_percentage))
team_win_percentage.to_csv('../data/team_level_data.csv', index=False)

# %%
print("Loading advanced box scores")
advanced_box_scores= pd.read_csv('../data/advanced_box_scores.csv')
games_to_get = games[~games['GAME_ID'].isin(advanced_box_scores['gameId'].unique())]['GAME_ID'].unique()
loading_bar = tqdm.tqdm(total=len(games_to_get))
skipped_games = []
for game_id in games_to_get:
    game_id = "00" + str(game_id)
    tries = 0
    while tries < 5:
        try:
            game = nba.boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id).get_data_frames()[0]
            time.sleep(0.6)
            break
        except:
            tries += 1
            time.sleep(0.6)
    if tries == 5:
        print('Skipping game: {}'.format(game_id))
        skipped_games.append(game_id)
    # Append game data to csv
    else:
        game.to_csv('../data/advanced_box_scores.csv', mode='a', header=False, index=False)
    loading_bar.update(1)

# %%
print("Loading traditional box scores")
traditional_box_scores = pd.read_csv('../data/games_details.csv')
games_to_get = games[~games['GAME_ID'].isin(traditional_box_scores['GAME_ID'].unique())]['GAME_ID'].unique()
loading_bar = tqdm.tqdm(total=len(games_to_get))
skipped_games = []
for game_id in games_to_get:
    game_id = "00" + str(game_id)
    tries = 0
    while tries < 5:
        try:
            game = nba.boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).get_data_frames()[0]
            break
        except:
            tries += 1
            time.sleep(0.6)
    if tries == 5:
        print('Skipping game: {}'.format(game_id))
        skipped_games.append(game_id)
    # Append game data to csv
    else:
        game.to_csv('../data/games_details.csv', mode='a', header=False, index=False)
    loading_bar.update(1)# Get defensive box scores

# Only use games from the 2017-18 season and later for the wide dataset
games = games[games['SEASON_ID'] >= 22017]

print("Loading defensive box scores")
defensive_box_scores = pd.read_csv('../data/defensive_box_scores.csv')
games_to_skip = pd.read_csv('../data/games_to_skip.csv')
games_to_get = games[~games['GAME_ID'].isin(defensive_box_scores['gameId'].unique())]['GAME_ID']
games_to_get = games_to_get[~games_to_get.isin(games_to_skip['GAME_ID'])].unique()
# games_to_get = games['GAME_ID'].unique()
for game_id in tqdm.tqdm(games_to_get):
    game_id = "00" + str(game_id)
    tries = 0
    while tries < 5:
        try:
            game = nba.boxscoredefensivev2.BoxScoreDefensiveV2(game_id=game_id).get_data_frames()[0]
            break
        except:
            tries += 1
            time.sleep(0.6)
    if tries == 5:
        print('Skipping game: {}'.format(game_id))
    # Append game data to csv
    else:
        game.to_csv('../data/defensive_box_scores.csv', mode='a', header=False, index=False)

# %%
# Get usage stats
print("Loading usage stats")
usage_stats = pd.read_csv('../data/usage_stats.csv')
games_to_get = games[~games['GAME_ID'].isin(usage_stats['GAME_ID'].unique())]['GAME_ID'].unique()
for game_id in tqdm.tqdm(games_to_get):
    game_id = "00" + str(game_id)
    tries = 0
    while tries < 5:
        try:
            game = nba.boxscoreusagev2.BoxScoreUsageV2(game_id=game_id).get_data_frames()[0]
            break
        except:
            tries += 1
            time.sleep(0.6)
    if tries == 5:
        print('Skipping game: {}'.format(game_id))
        skipped_games.append(game_id)
    # Append game data to csv
    else:
        game.to_csv('../data/usage_stats.csv', mode='a', header=False, index=False)

# Get hustle box scores
print("Loading hustle stats")
hustle_box_scores = pd.read_csv('../data/hustle_box_scores.csv')
games_to_get = games[~games['GAME_ID'].isin(hustle_box_scores['gameId'].unique())]['GAME_ID'].unique()
# games_to_get = games['GAME_ID'].unique()
for game_id in tqdm.tqdm(games_to_get):
    game_id = "00" + str(game_id)
    tries = 0
    while tries < 5:
        try:
            game = nba.boxscorehustlev2.BoxScoreHustleV2(game_id=game_id).get_data_frames()[0]
            break
        except:
            tries += 1
            time.sleep(0.6)
    if tries == 5:
        print('Skipping game: {}'.format(game_id))
    # Append game data to csv
    else:
        game.to_csv('../data/hustle_box_scores.csv', mode='a', header=False, index=False)

# Get scoring box scores
print("Loading scoring stats")
scoring_box_scores = pd.read_csv('../data/scoring_box_scores.csv')
games_to_get = games[~games['GAME_ID'].isin(scoring_box_scores['GAME_ID'].unique())]['GAME_ID'].unique()
# games_to_get = games['GAME_ID'].unique()
for game_id in tqdm.tqdm(games_to_get):
    game_id = "00" + str(game_id)
    tries = 0
    while tries < 5:
        try:
            game = nba.boxscorescoringv2.BoxScoreScoringV2(game_id=game_id).get_data_frames()[0]
            break
        except:
            tries += 1
            time.sleep(0.6)
    if tries == 5:
        print('Skipping game: {}'.format(game_id))
    # Append game data to csv
    else:
        game.to_csv('../data/scoring_box_scores.csv', mode='a', header=False, index=False)

season_dates = {
    '2005-2006': { 'start': date(2005, 11, 1), 'end': date(2006, 6, 20) },
    '2006-2007': { 'start': date(2006, 10, 31), 'end': date(2007, 6, 14) },
    '2007-2008': { 'start': date(2007, 10, 30), 'end': date(2008, 6, 17) },
    '2008-2009': { 'start': date(2008, 10, 28), 'end': date(2009, 6, 14) },
    '2009-2010': { 'start': date(2009, 10, 27), 'end': date(2010, 6, 17) },
    '2010-2011': { 'start': date(2010, 10, 26), 'end': date(2011, 6, 12) },
    '2011-2012': { 'start': date(2011, 12, 25), 'end': date(2012, 6, 21) },
    '2012-2013': { 'start': date(2012, 10, 30), 'end': date(2013, 6, 20) },
    '2013-2014': { 'start': date(2013, 10, 29), 'end': date(2014, 6, 15) },
    '2014-2015': { 'start': date(2014, 10, 28), 'end': date(2015, 6, 16) },
    '2015-2016': { 'start': date(2015, 10, 27), 'end': date(2016, 6, 19) },
    '2016-2017': { 'start': date(2016, 10, 25), 'end': date(2017, 6, 12) },
    '2017-2018': { 'start': date(2017, 10, 17), 'end': date(2018, 6, 8) },
    '2018-2019': { 'start': date(2018, 10, 16), 'end': date(2019, 6, 13) },
    '2019-2020': { 'start': date(2019, 10, 22), 'end': date(2020, 10, 11) },
    '2020-2021': { 'start': date(2020, 12, 22), 'end': date(2021, 7, 22) },
    '2021-2022': { 'start': date(2021, 10, 19), 'end': date(2022, 6, 16) },
    '2022-2023': { 'start': date(2022, 10, 18), 'end': date(2023, 6, 15) },
    '2023-2024': { 'start': date(2023, 10, 17), 'end': date(2024, 6, 13) },
}

numeric_columns = ['SEC', 'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M',
                   'FG3A', 'FG3_PCT',  'FTM', 'FTA', 'FT_PCT', 'OREB', 
                   'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF']

non_numeric_colums = ['GAME_ID', 'PLAYER_ID', 'SEASON', 'GAME_DATE', 'TEAM_ID']

v3_non_numeric_colums = ['teamCity', 'teamTricode', 'teamName', 'teamSlug', 'firstName', 'familyName', 'nameI', 'playerSlug', 'position', 'comment', 'jerseyNum', 'minutes']
defensive_box_scores_non_numeric = ['teamCity', 'teamTricode', 'teamName', 'teamSlug', 'firstName', 'familyName', 'nameI', 'playerSlug', 'position', 'comment', 'jerseyNum', 'matchupMinutes']

games = pd.read_csv('../data/games.csv')

# Convert GAME_DATE to datetime
games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])

# Sort by GAME_DATE
games.sort_values('GAME_DATE', inplace=True)

# Extract game ids
game_ids = games['GAME_ID'].unique()

# Load games_details
games_details_deep = pd.read_csv('../data/games_details.csv')

print("Pre filter")
print(games_details_deep.head())

# Keep only games in game_ids
games_details_deep = games_details_deep[games_details_deep['GAME_ID'].isin(game_ids)]
games_details_deep = games_details_deep[games_details_deep['COMMENT'].isna()]

# Drop unnecessary columns
games_details_deep.drop(['PLAYER_NAME', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'START_POSITION', 'NICKNAME', 'COMMENT'], axis=1, inplace=True)
print("Post filter")
print(games_details_deep.head())

# Load advanced_box_scores
advanced_box_scores = pd.read_csv('../data/advanced_box_scores.csv')
advanced_box_scores = advanced_box_scores[advanced_box_scores['comment'].isna()]
advanced_box_scores.drop(v3_non_numeric_colums, axis=1, inplace=True)
advanced_box_scores.rename(columns={'gameId': 'GAME_ID', 'teamId': 'TEAM_ID', 'personId': 'PLAYER_ID'}, inplace=True)
advanced_box_scores = advanced_box_scores[advanced_box_scores['GAME_ID'].isin(game_ids)]

# Load defensive_box_scores
defensive_box_scores = pd.read_csv('../data/defensive_box_scores.csv')
defensive_box_scores = defensive_box_scores[defensive_box_scores['comment'].isna()]
defensive_box_scores.drop(defensive_box_scores_non_numeric, axis=1, inplace=True)
defensive_box_scores.rename(columns={'gameId': 'GAME_ID', 'teamId': 'TEAM_ID', 'personId': 'PLAYER_ID'}, inplace=True)
defensive_box_scores = defensive_box_scores[defensive_box_scores['GAME_ID'].isin(game_ids)]

# Load usage stats
usage_stats = pd.read_csv('../data/usage_stats.csv')
usage_stats = usage_stats[usage_stats['COMMENT'].isna()]
usage_stats.drop(['MIN', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'COMMENT', 'PLAYER_NAME', 'NICKNAME', 'START_POSITION'], axis=1, inplace=True)
usage_stats = usage_stats[usage_stats['GAME_ID'].isin(game_ids)]

#Load hustle stats
hustle_stats = pd.read_csv('../data/hustle_box_scores.csv')
hustle_stats = hustle_stats[hustle_stats['comment'].isna()]
hustle_stats.drop(v3_non_numeric_colums, axis=1, inplace=True)
hustle_stats.rename(columns={'gameId': 'GAME_ID', 'teamId': 'TEAM_ID', 'personId': 'PLAYER_ID'}, inplace=True)
hustle_stats = hustle_stats[hustle_stats['GAME_ID'].isin(game_ids)]

#Load scoring stats
scoring_stats = pd.read_csv('../data/scoring_box_scores.csv')
scoring_stats = scoring_stats[scoring_stats['COMMENT'].isna()]
scoring_stats.drop(['MIN', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'COMMENT', 'PLAYER_NAME', 'NICKNAME', 'START_POSITION'], axis=1, inplace=True)
scoring_stats = scoring_stats[scoring_stats['GAME_ID'].isin(game_ids)]


# Add game date to games_details
games_details_deep = games_details_deep.merge(games.groupby('GAME_ID').first()['GAME_DATE'], on='GAME_ID', how='left')

print("Post game_date merge")
print(games_details_deep.head())

# Drop rows with missing values (Player did not play)
games_details_deep.dropna(inplace=True)

# Convert MIN to seconds
def convert_to_seconds(x):
    x = x.split(':')
    return int(float(x[0]))*60 + int(float(x[1])) if len(x) == 2 else int(float(x[0]))*60

games_details_deep['MIN'] = games_details_deep['MIN'].apply(convert_to_seconds)
games_details_deep.rename(columns={'MIN': 'SEC'}, inplace=True)

# Add a column for the season that the game was played in
def get_season(x):
    x = x.date()
    for key, value in season_dates.items():
        if x >= value['start'] and x <= value['end']:
            return key
    return None

games_details_deep['SEASON'] = games_details_deep['GAME_DATE'].apply(get_season)
games_details_wide = games_details_deep.copy()

games_details_deep = games_details_deep[games_details_deep['GAME_ID'].isin(advanced_box_scores['GAME_ID'])]
games_details_deep = games_details_deep[games_details_deep['PLAYER_ID'].isin(advanced_box_scores['PLAYER_ID'].unique())]

print("Pre merge")
print(games_details_deep.head())

games_details_deep = games_details_deep.merge(advanced_box_scores, on=['GAME_ID', 'TEAM_ID', 'PLAYER_ID'], how='left')

games_details_wide = games_details_deep[games_details_deep['GAME_ID'].isin(advanced_box_scores['GAME_ID'])]
games_details_wide = games_details_deep[games_details_deep['PLAYER_ID'].isin(advanced_box_scores['PLAYER_ID'].unique())]
games_details_wide = games_details_wide[games_details_wide['GAME_ID'].isin(defensive_box_scores['GAME_ID'])]
games_details_wide = games_details_wide[games_details_wide['PLAYER_ID'].isin(defensive_box_scores['PLAYER_ID'].unique())]
games_details_wide = games_details_wide[games_details_wide['GAME_ID'].isin(usage_stats['GAME_ID'])]
games_details_wide = games_details_wide[games_details_wide['PLAYER_ID'].isin(usage_stats['PLAYER_ID'].unique())]
games_details_wide = games_details_wide[games_details_wide['GAME_ID'].isin(hustle_stats['GAME_ID'])]
games_details_wide = games_details_wide[games_details_wide['PLAYER_ID'].isin(hustle_stats['PLAYER_ID'].unique())]
games_details_wide = games_details_wide[games_details_wide['GAME_ID'].isin(scoring_stats['GAME_ID'])]
games_details_wide = games_details_wide[games_details_wide['PLAYER_ID'].isin(scoring_stats['PLAYER_ID'].unique())]

games_details_wide = games_details_deep.merge(advanced_box_scores, on=['GAME_ID', 'TEAM_ID', 'PLAYER_ID'], how='left')
games_details_wide = games_details_wide.merge(usage_stats, on=['GAME_ID', 'TEAM_ID', 'PLAYER_ID'], how='left')
games_details_wide = games_details_wide.merge(defensive_box_scores, on=['GAME_ID', 'TEAM_ID', 'PLAYER_ID'], how='left')
games_details_wide = games_details_wide.merge(hustle_stats, on=['GAME_ID', 'TEAM_ID', 'PLAYER_ID'], how='left')
games_details_wide = games_details_wide.merge(scoring_stats, on=['GAME_ID', 'TEAM_ID', 'PLAYER_ID'], how='left')

games_details_deep.dropna(inplace=True)
games_details_wide.dropna(inplace=True)

print("Post merge")
print(games_details_deep.head())

# Sort by GAME_DATE, GAME_ID, TEAM_ID, and MIN
games_details_deep.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID', 'SEC'], ascending=[True, True, True, False], inplace=True)
games_details_deep.reset_index(drop=True, inplace=True)

games_details_wide.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID', 'SEC'], ascending=[True, True, True, False], inplace=True)
games_details_wide.reset_index(drop=True, inplace=True)

# Create a new dataframe and store the rolling averages of the past 10 games in each season for each player
window = 10
min_periods = 10
rolling_averages_deep = pd.DataFrame()
numeric_columns = games_details_deep.columns.drop(non_numeric_colums)
for season in season_dates.keys():
    season_games_details = games_details_deep[games_details_deep['SEASON'] == season]
    for team in season_games_details['TEAM_ID'].unique():
        team_games_details = season_games_details[season_games_details['TEAM_ID'] == team].sort_values('GAME_DATE')
        for player in team_games_details['PLAYER_ID'].unique():
            player_games_details = team_games_details[team_games_details['PLAYER_ID'] == player].sort_values('GAME_DATE')
            player_games_details.reset_index(drop=True, inplace=True)
            # For each numeric column, calculate the rolling average of the past 10 games
            for column in numeric_columns:
                player_games_details.loc[:, column] = player_games_details.loc[:, column]
                player_games_details.loc[:, column] = player_games_details[column].rolling(window=window, min_periods=min_periods).mean()
            # Make column for number of days since last game
            player_games_details.loc[:, 'DAYS_SINCE_LAST_GAME'] = player_games_details['GAME_DATE'].diff().dt.days
            # Shift the game id and game date columns up by 1
            player_games_details.loc[:, 'GAME_ID'] = player_games_details['GAME_ID']
            player_games_details.loc[:, 'GAME_DATE'] = player_games_details['GAME_DATE'].shift(-1)
            player_games_details.loc[:, 'DAYS_SINCE_LAST_GAME'] = player_games_details['DAYS_SINCE_LAST_GAME'].shift(-1)
            rolling_averages_deep = pd.concat([rolling_averages_deep, player_games_details], ignore_index=True)
rolling_averages_deep.dropna(inplace=True)
# Sort by GAME_DATE, GAME_ID, TEAM_ID, and SEC
rolling_averages_deep.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID', 'SEC'], ascending=[True, True, True, False], inplace=True)
rolling_averages_deep.reset_index(drop=True, inplace=True)# Standardize the data
print(rolling_averages_deep.head())
rolling_averages_deep.to_csv('../data/rolling_averages_deep.csv', index=False)

window = 10
min_periods = 10
rolling_averages_wide = pd.DataFrame()
numeric_columns = games_details_wide.columns.drop(non_numeric_colums)

for season in season_dates.keys():
    season_games_details = games_details_wide[games_details_wide['SEASON'] == season]
    for team in season_games_details['TEAM_ID'].unique():
        team_games_details = season_games_details[season_games_details['TEAM_ID'] == team].sort_values('GAME_DATE')
        for player in team_games_details['PLAYER_ID'].unique():
            player_games_details = team_games_details[team_games_details['PLAYER_ID'] == player].sort_values('GAME_DATE')
            player_games_details.reset_index(drop=True, inplace=True)
            # For each numeric column, calculate the rolling average of the past 10 games
            for column in numeric_columns:
                player_games_details.loc[:, column] = player_games_details[column].rolling(window=window, min_periods=min_periods).mean()
            # Make column for number of days since last game
            player_games_details.loc[:, 'DAYS_SINCE_LAST_GAME'] = player_games_details['GAME_DATE'].diff().dt.days
            # Shift the game id and game date columns up by 1
            player_games_details.loc[:, 'GAME_ID'] = player_games_details['GAME_ID'].shift(-1)
            player_games_details.loc[:, 'GAME_DATE'] = player_games_details['GAME_DATE'].shift(-1)
            player_games_details.loc[:, 'DAYS_SINCE_LAST_GAME'] = player_games_details['DAYS_SINCE_LAST_GAME'].shift(-1)
            rolling_averages_wide = pd.concat([rolling_averages_wide, player_games_details], ignore_index=True)
rolling_averages_wide.dropna(inplace=True)
# Sort by GAME_DATE, GAME_ID, TEAM_ID, and SEC
rolling_averages_wide.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID', 'SEC'], ascending=[True, True, True, False], inplace=True)
rolling_averages_wide.reset_index(drop=True, inplace=True)# Standardize the data
print(rolling_averages_wide.head())
rolling_averages_wide.to_csv('../data/rolling_averages_wide.csv', index=False)
