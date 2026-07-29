
from pprint import pprint
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from dataset import PolicyFeatureDataset
from vocab import OptionsVocab, SelectionVocab


@dataclass
class DecisionContextVocab:
    options: OptionsVocab
    selection: SelectionVocab

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
pprint(dataset[79])
print("==="*50)
print(f"decision length: {len(dataset[1][0]['features']['decision_chain'])}")
