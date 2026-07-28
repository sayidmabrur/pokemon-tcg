
from pprint import pprint
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from dataset import PolicyFeatureDataset

@dataclass
class OptionsVocab:
    area: int
    attackId: int
    count: float
    energyIndex: int


    pass

@dataclass
class SelectionVocab:
    pass

@dataclass
class DecisionContextVocab:
    options: List[OptionsVocab]
    Selection: List[SelectionVocab]

    pass

class PolicyNetwork(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, obs):
        pass


policy = PolicyNetwork()


dataset = PolicyFeatureDataset(
    "data/policy_decisions.parquet", player_name="Yushin Ito")
print("dataset sample:")
print("==="*50)
pprint(dataset[0])
print("==="*50)
print(f"decision length: {len(dataset[9][0]['features']['decision_chain'])}")
