from abc import ABC, abstractmethod

import torch.nn as nn


class PolicyInference(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def act(self, obs):
        raise NotImplementedError
