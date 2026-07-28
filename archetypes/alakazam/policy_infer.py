def Policy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, obs):
        pass

def PolicyInference(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def act(self, obs):
        raise NotImplementedError