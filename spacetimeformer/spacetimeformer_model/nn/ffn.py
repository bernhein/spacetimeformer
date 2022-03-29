import torch.nn as nn
import torch

class FeedForwardNetwork(nn.Module):
    def __init__(self,input:int, hidden:int, output):
        super(FeedForwardNetwork, self).__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential( #sequential operation
            nn.Linear(input, input), 
            nn.Sigmoid(), 
            nn.Linear(input, output), 
            nn.Sigmoid(),
        )

    def forward(self, X):
        return self.net(X)