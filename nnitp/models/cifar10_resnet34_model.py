#import numpy

#from tensorflow.keras.models import load_model
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from nnitp.models.resnet import resnet34
from nnitp.model_wrapper import Wrapper
#from nnitp.models.models import CIFAR10_Model
#from model_wrapper import Wrapper
#from nnitp.datatype import Image

# Fetch the CIFAR10 dataset, normalized with mean=0, std = 1


def get_data():
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
    return train_data, test_data

# Fetch the trained model

def get_model():
    model = resnet34(num_classes = 10)
    pretraineddir = os.environ["PRETRAINEDDIR"]
    modeldir = os.path.join(pretraineddir, "resnet34_cifar10.pth")
    model.load_state_dict(torch.load(modeldir))
    return Wrapper(model, [1,3,32,32])

params = {'size':20000,'alpha':0.95,'gamma':0.6,'mu':0.9,'layers':[50]}

