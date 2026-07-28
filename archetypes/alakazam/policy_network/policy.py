
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from dataset import PolicyFeatureDataset

class PolicyNetwork(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, obs):
        pass


policy = PolicyNetwork()

replays_dir = "replays/benarg/"

dataset = PolicyFeatureDataset(replays_dir)

print("dataset sample:", dataset[0])