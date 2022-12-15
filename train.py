import numpy as np
from utils import OrthoLoss, KeepTrack
import conf as cfg
from datasetup import dataset, create_dataloader_split
from model import OrthoSource
import engine
import argparse
import torch
from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(prog='train.py', description='required flags and supplemtary parameters for training')
parser.add_argument('--train', action=argparse.BooleanOptionalAction)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
# parser.add_argument('--val', action=argparse.BooleanOptionalAction)
parser.add_argument('--epoch', '-e', type=int, required=False, metavar='epoch', default=1)

args = parser.parse_args()


def train(net, train_loader, val_loader, opt, criterion, epochs, minerror, modelname:str):
    kt = KeepTrack(path=cfg.paths['model'])
    for epoch in range(epochs):
        train_loss = engine.train_step(model=net, data=train_loader, criterion=criterion, optimizer=opt)
        val_loss = engine.val_step(model=net, data=val_loader, criterion=criterion)
        if val_loss < minerror:
            val_loss = minerror
            kt.save_ckp(model=net, opt=opt, epoch=epoch, minerror=val_loss, fname=modelname)
            
        print(f"train_loss={train_loss} val_loss={val_loss}")
    


def main():
    model_name = f"orthosource.pt"
    keeptrack = KeepTrack(path=cfg.paths['model'])
    Net = OrthoSource()
    opt = optim.Adam(params=Net.parameters(), lr=3e-4)
    criteria = OrthoLoss()
    train_data, val_data, test_data = create_dataloader_split(dataset=dataset)
    minerror = np.inf
    if args.train:
        train(net=Net, train_loader=train_data, val_loader=val_data, opt=opt, criterion=criteria, epochs=args.epoch, minerror=minerror, modelname=model_name)
    
    if args.test:
        state = keeptrack.load_ckp(fname=model_name)
        Net.load_state_dict(state['model'])
        print(f"min error is {state['minerror']} which happen at epoch {state['epoch']}")
        engine.test_step(model=Net)
        


if __name__ == '__main__':
    main()