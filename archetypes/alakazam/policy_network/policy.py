
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from dataset import PolicyFeatureDataset

from dataset import PolicyFeatureDataset

class PolicyNetwork(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, obs):
        pass


policy = PolicyNetwork()

from pprint import pprint

dataset = PolicyFeatureDataset("data/policy_decisions.parquet", player_name="Yushin Ito")
print("dataset sample:")
print("==="*50)
pprint( dataset[0])
print("==="*50)
print(f"decision length: {len(dataset[0][0]['features']['decision_chain'])}")