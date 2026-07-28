
from abc import ABC, abstractmethod


class State(ABC):

    def __init__(self, deck_csv_path: str, name: str, description: str):
        self.deck_csv_path = deck_csv_path
        self.name = name
        self.description = description

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        raise NotImplementedError
