import torch
import torch.nn as nn
from torchvision import datasets, transforms

class Flatten(nn.Module):

  def __init__(self):
      super(Flatten, self).__init__()

  def forward(self, x):
      #print(x.view(x.size(0), -1).shape)
      return x.view(x.size(0), -1)


class CIFAR10_Model(torch.nn.Module):
  def __init__(self):
    super(CIFAR10_Model, self).__init__()
    self.feature = nn.Sequential(
      nn.Conv2d(3, 32, 3, 1, padding = 1),
      nn.ELU(),
      nn.BatchNorm2d(32),
      nn.Conv2d(32, 32, 3, 1,padding = 1),
      nn.ELU(),
      nn.BatchNorm2d(32),
      nn.MaxPool2d(2),
      nn.Dropout(0.2),
      nn.Conv2d(32, 64, 3, 1,padding=1),
      nn.ELU(),
      nn.BatchNorm2d(64),
      nn.Conv2d(64, 64, 3, 1,padding = 1),
      nn.ELU(),
      nn.BatchNorm2d(64),
      nn.MaxPool2d(2),
      nn.Dropout(0.3),
      nn.Conv2d(64, 128, 3, 1,padding = 1),
      nn.ELU(),
      nn.BatchNorm2d(128),
      nn.Conv2d(128, 128, 3, 1,padding=1),
      nn.ELU(),
      nn.BatchNorm2d(128),
      nn.MaxPool2d(2),
      nn.Dropout(0.4),
    )


    self.classifier = nn.Sequential(
      Flatten(),
      nn.Linear(2048, 10),
    )



  def forward(self, x):
    x = self.feature(x)
    x = self.classifier(x)
    return x



class MNIST_Model(torch.nn.Module):
  def __init__(self):
    super(MNIST_Model, self).__init__()
    self.feature = nn.Sequential(
      nn.Conv2d(1, 28, 3, 1),
      nn.MaxPool2d(2),
    )


    self.classifier = nn.Sequential(
      Flatten(),
      nn.Linear(4732, 128),
      nn.Dropout(0.2),
      nn.Linear(128, 10),
    )



  def forward(self, x):
    x = self.feature(x)
    x = self.classifier(x)
    return x



