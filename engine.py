import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import numpy as np

def train_step(model: nn.Module, data: DataLoader, criterion: nn.Module, optimizer: optim):
    epoch_error = 0
    l = len(data)
    model.train()
    for i, (X, Y) in enumerate(data):
        out = model(X)
        loss = criterion(out, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_error += loss.item()

    return epoch_error/l


def val_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    with torch.no_grad():
        for i, (X, Y) in enumerate(data):
            out = model(X)
            loss = criterion(out, Y)
            epoch_error += loss.item()

    return epoch_error/l


def test_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    with torch.no_grad():
        for i, (X, Y) in enumerate(data):
            out = model(X)
            loss = criterion(out, Y)
            epoch_error += loss.item()

    print(f"test-loss={epoch_error}")

    X = out.numpy()
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random', perplexity=30).fit_transform(X)
    print(X_embedded.shape)




def main():
    pass



if __name__ == '__main__':
    main()