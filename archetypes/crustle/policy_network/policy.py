
from pprint import pprint
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from dataset import PolicyFeatureDataset, transform
from vocab import OptionsVocab, SelectionVocab

class DecisionChainEncoder(nn.Module):
    pass

class PlayerStateEncoder(nn.Module):
    pass

class DecisionContextEncoder(nn.Module):
    pass

class OpponentStateEncoder(nn.Module):
    pass

class PlayerStateEncoder(nn.Module):
    pass

class OpponentHistoryEncoder(nn.Module):
    pass

class GlobalStateEncoder(nn.Module):
    pass


class PolicyNetworkEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, obs):
        pass


policy = PolicyNetworkEncoder()


dataset = PolicyFeatureDataset(
    "data/policy_decisions_crustle.parquet", player_name="flg", transform=transform)
print("dataset sample:")
print("===" * 50)
# pprint(dataset[79])
print("===" * 50)
# print(f"decision chain length: {len(dataset[79][0]['features']['decision_chain']['turn'])}")
print("View all keys:")
print(dataset[79][0]['features'].keys())