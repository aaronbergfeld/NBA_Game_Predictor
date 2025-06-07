import pytest

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from src.dataloaders import GamesDataset


def make_sample_dataset():
    # Two rows representing one game (home and away)
    games = pd.DataFrame({
        'GAME_ID': [1, 1],
        'TEAM_ID': [100, 200],
        'HOME_TEAM': [1, 0],
        'HOME_TEAM_WON': [1, 1]
    })
    # Ten players for each team
    players = []
    for team_id in [100, 200]:
        for i in range(10):
            players.append({
                'GAME_ID': 1,
                'TEAM_ID': team_id,
                'PLAYER_ID': team_id*10 + i,
                'GAME_DATE': '2024-01-01',
                'SEASON': '2023-2024',
                'SEC': 100 - i,
                'PTS': i
            })
    players = pd.DataFrame(players)
    team_stats = pd.DataFrame({
        'TEAM_ID': [100, 200],
        'GAME_ID': [1, 1],
        'STAT': [5, 7]
    })
    odds = pd.DataFrame({
        'GAME_ID': [1],
        'best_home_odds': [1.5],
        'best_away_odds': [2.5]
    })
    return games, players, team_stats, odds


def test_games_dataset_length():
    games, players, team_stats, odds = make_sample_dataset()
    ds = GamesDataset(games, players, team_stats, odds)
    assert len(ds) == 1


def test_games_dataset_item_shapes():
    games, players, team_stats, odds = make_sample_dataset()
    ds = GamesDataset(games, players, team_stats, odds)
    x, team_data, label, home_odds, away_odds = ds[0]
    assert x.shape == (2, 10, 2)
    assert team_data.shape == (2, 1)
    assert isinstance(label, (int, float))
    assert home_odds == 1.5
    assert away_odds == 2.5
