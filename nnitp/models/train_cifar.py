import torch
import numpy as np
import copy
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)




class Flatten(nn.Module):

  def __init__(self):
      super(Flatten, self).__init__()

  def forward(self, x):
      #print(x.view(x.size(0), -1).shape)
      return x.view(x.size(0), -1)



class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
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







def train(model, device, train_loader, optimizer, epoch):
    model.train()
    crit = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = crit(output, target)

        loss.backward()
        optimizer.step()
        #test(model,device,test_loader)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = crit(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))





if __name__ == "__main__":
  np.random.seed(123)
  torch.manual_seed(123)
  torch.cuda.manual_seed(123)


  #transform = transforms.Compose([
  #  transforms.ToTensor(),
  #  transforms.Normalize((0.1307,),(0.3081,))
  #])

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
          (0.2023, 0.1994, 0.2010)),
   ])

  test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
    ])


  train_data = datasets.CIFAR10('data', train = True, download = True, transform = train_transform)
  test_data = datasets.CIFAR10('data', train = False, download = True, transform = test_transform)

  print(len(train_data))
  print(len(test_data))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True, num_workers = 4, pin_memory = True)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64,  shuffle = True, num_workers = 4, pin_memory = True)

  model = Model().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.8)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 70,0)


  for epoch in range(1,70):
      train(model, device, train_loader, optimizer, epoch)
      test(model, device, test_loader)
      scheduler.step()

  torch.save(model,"cifar_model.pt")






