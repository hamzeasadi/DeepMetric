import cv2 as cv
import numpy as np
import os, sys
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
import conf as cfg
from torchvision import models
from torch.utils.data import random_split, DataLoader

resnet_weight = models.ResNet50_Weights.DEFAULT

auto_transform = resnet_weight.transforms()

train_transform = transforms.Compose([
    transforms.Resize(224),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    # transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

dataset = ImageFolder(root=cfg.paths['libbherr'], transform=auto_transform)

def create_dataloader_split(dataset, train_percent=0.8, batch_size=16):
    data_size = len(dataset)
    train_size = int(train_percent*data_size)
    evaluation_size = data_size - train_size
    validation_size = int(evaluation_size*0.75)
    test_size = evaluation_size - validation_size
    train, evaluation = random_split(dataset=dataset, lengths=[train_size, evaluation_size])
    validation, test = random_split(dataset=evaluation, lengths=[validation_size, test_size])
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size)

    return train_loader, validation_loader, test_loader
    
    

def main():
    # print(dataset.class_to_idx)
    # loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    # first_batch = next(iter(loader))
    # print(first_batch[0][0].shape)
    # plt.imshow(first_batch[0][0].permute(1,2,0), cmap='gray')
    # plt.show()
    print(auto_transform)



if __name__ == '__main__':
    main()