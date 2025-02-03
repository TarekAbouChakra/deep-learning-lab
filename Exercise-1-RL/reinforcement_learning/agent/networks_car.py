import torch.nn as nn
import torch
import torch.nn.functional as F

# Network Created for car race DQN, I made the number of parameter smaller to train,
# so that it can train faster

class CNN(nn.Module):
    def __init__(self, action_dim, history_length = 0) -> None:
        super(CNN, self).__init__()

        self.pool    = nn.MaxPool2d(2, 2)
        self.relu    = nn.ReLU()
        self.conv1   = nn.Conv2d(in_channels=history_length+1, out_channels=6, kernel_size=(6, 6), stride=2) 
        self.conv2   = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5), stride=2)
        self.fc1     = nn.Linear(12*5*5, 216)
        self.fc2     = nn.Linear(216, 108) 
        self.fc3     = nn.Linear(108, action_dim) 

    def forward(self, x):
        # TODO: compute forward pass
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(-1, 12*5*5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x