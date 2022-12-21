import torch
import torchvision
import os
import cv2
from torch import nn as nn
from torchvision import models
from torchinfo import summary
from datasetup import dataset, create_dataloader_split
from utils import get_triplet_mask, OrthoLoss
from torch.optim import Adam



class Identity(nn.Module):
    """
    doc
    """
    def __init__(self) -> None:
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x





class OrthoSource(nn.Module):
    """
    doc
    """
    def __init__(self) -> None:
        super().__init__()
        resnet_weight = models.ResNet50_Weights.DEFAULT
        self.base_model = models.resnet50(weights=resnet_weight)
        # self.base_model.avgpool = Identity()
        self.base_model.fc = Identity()
        

        for param in self.base_model.parameters():
            param.requires_grad = False
        
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        # for param in self.base_model.fc.parameters():
        #     param.requires_grad = True

        # self.base_model.fc = nn.Linear(in_features=2048, out_features=1000)


    def forward(self, x):
        x = self.base_model(x)
        return x 


    


def main():
    model = OrthoSource()
    x = torch.randn(size=(1, 3, 240, 240))
    summary(model=model, input_size=(10, 3, 224, 224), col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
    row_settings=["var_names"])
    # for param in model.parameters():
    # print(model.state_dict().keys())
    # for name, layer in model.named_modules():
    #     print(name)

    # # print(model.base_model.layer4)
    # criterion = OrthoLoss()
    # opt = Adam(params=model.parameters(), lr=0.001)
    # train, val, test = create_dataloader_split(dataset=dataset)
    # for batch in test:
    #     out = model(batch[0])
    #     loss = criterion(out, batch[1])
    #     # opt.zero_grad()
    #     # loss.backward()
    #     # opt.step()

    #     print(f"loss={loss.item()}")
    #     break

if __name__ == '__main__':
    main()