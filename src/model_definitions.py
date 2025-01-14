import torch
from torch import nn

class PlayerDataAggregator(nn.Module):
    def __init__(self, num_players):
        super(PlayerDataAggregator, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=(num_players, 1))
        
    def forward(self, x):
        x = self.conv1(x)
        return x

    
class LogisticRegression(nn.Module):
    def __init__(self, num_players, num_features):
        super(LogisticRegression, self).__init__()
        self.player_data_aggregator = PlayerDataAggregator(num_players)
        self.linear = nn.Linear(num_features, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.player_data_aggregator(x))
        x = self.sigmoid(self.linear(x))
        return x

    
class NeuralNetwork(nn.Module):
    def __init__(self, num_players, num_features, num_team_features, hidden_layers=[16, 4]):
        super(NeuralNetwork, self).__init__()
        self.player_data_aggregator = PlayerDataAggregator(num_players)
        self.linear1 = nn.Linear(2*num_features + 2*num_team_features, hidden_layers[0])
        self.linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.linear3 = nn.Linear(hidden_layers[1], 1)
        self.relu = nn.ReLU()
    
    def forward(self, x_player, x_team):
        x = self.relu(self.player_data_aggregator(x_player))
        # Flatten data
        x = x.view(x.size(0), -1)
        x_team = x_team.view(x_team.size(0), -1)
        x = torch.cat((x, x_team), dim=1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
import torch
import torch.nn as nn
import torch.optim as optim

import torch
from torch import nn, optim
from torch.nn.functional import sigmoid
from scipy.optimize import minimize
import numpy as np

# Assuming `model` is your pre-trained model

class TemperatureScaledModel(nn.Module):
    def __init__(self, base_model):
        super(TemperatureScaledModel, self).__init__()
        self.base_model = base_model
        # Initialize temperature to 1 (no scaling initially)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, matchup, team_data):
        logits = self.base_model(matchup, team_data)
        # Scale logits by temperature
        return logits / self.temperature

    def set_temperature(self, validation_dataloader):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for matchup, team_data, outcome in validation_dataloader:
                matchup = matchup.to(dtype=torch.float32, device=device)
                team_data = team_data.to(dtype=torch.float32, device=device)
                outcome = outcome.view(-1, 1).to(dtype=torch.double, device=device)
                logits = self.base_model(matchup, team_data)
                logits_list.append(logits.cpu().numpy())
                labels_list.append(outcome.cpu().numpy())
        
        logits = np.concatenate(logits_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        def ece_loss_fn(temperature):
            predictions = np.zeros((len(logits), 2))
            predictions[:, 0] = (sigmoid(torch.tensor(logits) / temperature)).numpy().flatten()
            predictions[:, 1] = 1 - predictions[:, 0]
            ece = class_specific_ece(labels, predictions, n_classes=2, n_bins=20, min_bins_filled=0)
            return ece

        # Optimize the temperature parameter
        res = minimize(ece_loss_fn, x0=[1.0], bounds=[(0.1, 10.0)])
        optimized_temperature = res.x[0]
        print(f"Optimized Temperature: {optimized_temperature}")
        self.temperature.data = torch.tensor([optimized_temperature]).to(device)
        
class BrierScoreLoss(torch.nn.Module):
    def __init__(self):
        super(BrierScoreLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.sum((y_pred - y_true)**2, dim=1))

class LogisticRegressionMetaModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionMetaModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Input_dim is the number of features from base models
        
    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)  # Sigmoid to convert to probability
        return x

import numpy as np

def class_specific_ece(y_true, y_pred, n_classes, n_bins=10, min_bins_filled=0.8):
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    dataset_size = len(y_true)

    for k in range(n_classes):
        bin_counts = np.zeros(n_bins)
        correct_counts = np.zeros(n_bins)
        prob_sums = np.zeros(n_bins)

        # Group predictions into bins
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            for j in range(dataset_size):
                if y_pred[j, k] > bin_lower and y_pred[j, k] <= bin_upper:
                    bin_counts[i] += 1
                    prob_sums[i] += y_pred[j, k]
                    if y_true[j] == k:
                        correct_counts[i] += 1

        # Check if minimum percentage of bins are filled
        if np.sum(bin_counts > 0) < min_bins_filled * n_bins:
            return 1 + 1/np.sum(bin_counts > 0)

        for j in range(n_bins):
            if bin_counts[j] > 0:
                avg_prob = (prob_sums[j] / bin_counts[j]) if bin_counts[j] > 0 else 0
                accuracy = correct_counts[j] / bin_counts[j]
                ece += np.abs(accuracy - avg_prob) * (bin_counts[j] / dataset_size)

    return ece / n_classes

def train_test_split(seasons, test_season):
    train = []
    test = []
    for season, games in seasons.items():
        if season == test_season:
            test += games
            break
        else:
            train += games
    return train, test

def expected_calibration_error(samples, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece