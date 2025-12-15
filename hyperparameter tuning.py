import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from test_tube import Experiment

# Neural Network Model Definition
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Training Function
def train(model, criterion, optimizer, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

# Evaluation Function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Objective Function for Hyperopt
def objective(hparams):
    # Load MNIST dataset
    train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=int(hparams['batch_size']), shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=int(hparams['batch_size']), shuffle=False)

    # Initialize model, criterion, and optimizer
    model = SimpleNN(784, int(hparams['hidden_size']), 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])

    # Train the model
    model = train(model, criterion, optimizer, train_loader, int(hparams['epochs']))

    # Evaluate the model
    accuracy = evaluate(model, test_loader)

    # Negative accuracy as the loss (since we want to maximize accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}

if __name__ == '__main__':

    # Hyperparameter Space
    space = {
        'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01),
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'hidden_size': hp.choice('hidden_size', [50, 100, 200]),
        'epochs': 5
    }

    # Running Hyperparameter Optimization
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

    print("Best hyperparameters:", best)
