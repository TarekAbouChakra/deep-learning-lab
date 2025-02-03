import torch
from agent.networks import CNN

#Please be sure to change the history length here when running one of the models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BCAgent:
    
    def __init__(self, learning_rate=0.001):
        # TODO: Define network, loss function, optimizer
        # self.net = CNN(...)
        self.net   = CNN(history_length=2, n_classes=5).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss  = torch.nn.CrossEntropyLoss()

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize
        
        X_batch = torch.tensor(X_batch).float().to(device)
        y_batch = torch.tensor(y_batch).long().to(device)

        output  = self.net(X_batch)

        loss = self.loss(output, y_batch)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss

    def predict(self, X):
        # TODO: forward pass
        X = torch.tensor(X).float().to(device)
        outputs = self.net(X)
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))


    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

