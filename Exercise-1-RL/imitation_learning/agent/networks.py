import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

# The code like this will run the CNN models, to include the LSTM please comment the self.fc1 in __init__
# and the output of x using fc1 in forward. Also remove comment from unsqueeze, lstm, and squeeze

class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=3): 
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.relu    = nn.ReLU()
        self.conv1   = nn.Conv2d(in_channels=history_length+1, out_channels=6, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1)
        self.conv3   = nn.Conv2d(in_channels=12, out_channels=16,kernel_size=3, padding=1)
        self.conv4   = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=3, padding=1)
        # self.lstm    = nn.LSTM(input_size=20*6*6, hidden_size=240, num_layers=1, batch_first=True)
        self.fc1     = nn.Linear(20*6*6, 240) 
        self.fc2     = nn.Linear(240, 120)
        self.fc3     = nn.Linear(120, 40)
        self.fc4     = nn.Linear(40, n_classes)

    def forward(self, x):
        # TODO: compute forward pass
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))

        x = x.view(-1, 20*6*6)

        # x = x.unsqueeze(1) 
        # x, _ = self.lstm(x)
        # x = x.squeeze(1) 

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

