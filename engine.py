import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from torchmetrics import Accuracy, ConfusionMatrix


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_step(model: nn.Module, data: DataLoader, criterion: nn.Module, optimizer: optim):
    epoch_error = 0
    l = len(data)
    model.train()
    for i, (X, Y) in enumerate(data):
        X, Y = X.to(dev), Y.to(dev)
        out = model(X)
        loss = criterion(out, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_error += loss.item()
        # break
    return epoch_error/l


def val_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    with torch.no_grad():
        for i, (X, Y) in enumerate(data):
            X, Y = X.to(dev), Y.to(dev)
            out = model(X)
            loss = criterion(out, Y)
            epoch_error += loss.item()
            # break
    return epoch_error/l


def test_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    with torch.no_grad():
        for i, (X, Y) in enumerate(data):
            out = model(X)
            # loss = criterion(out, Y)
            # epoch_error += loss.item()
  

    # y = Y.numpy()
    print(out.shape, Y.shape)
    
    yhat = torch.argmax(out, dim=1)
    acc = Accuracy(task='multiclass', num_classes=7)
    print(Y)
    print(yhat)
    accuracy = acc(yhat, Y)
    print(accuracy)
    # colors = ['green', 'blue', 'red', 'yellow', 'cyan', 'orange', 'black', 'magenta']
    # color_code = []
    # for i in range(len(y)):
    #     color_code.append(colors[y[i]])
    # # print(color_code)
    # # x = np.random.randint(low=1, high=10, size=(len(y), 2))
    # X = out.numpy()
    # X_embedded = TSNE(n_components=2, learning_rate='auto',
    #                 init='random', perplexity=3).fit_transform(X)
    # print(X_embedded.shape)

    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=color_code)
    # plt.savefig('output.png')
    # plt.show()



def main():
    pass



if __name__ == '__main__':
    main()