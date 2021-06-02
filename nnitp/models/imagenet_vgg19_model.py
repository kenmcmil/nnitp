#import numpy

#from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from nnitp.model_wrapper import Wrapper
from nnitp.models.vgg_imagenet import vgg19_bn
import os

# Fetch the CIFAR10 dataset, normalized with mean=0, std = 1




def get_data():
    imagedir = os.environ["IMAGENETDIR"]
    traindir = os.path.join(imagedir, "train")
    validdir = os.path.join(imagedir, "val")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    test_data = datasets.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    return train_data, test_data

# Fetch the trained model

def get_model():
    model = vgg19_bn()
    pretraineddir = os.environ["PRETRAINEDDIR"]
    modeldir = os.path.join(pretraineddir, "vgg19_imagenet.pth")
    model.load_state_dict(torch.load(modeldir))
    return Wrapper(model, [1,3,224,224])

params = {'size':5000,'alpha':0.85,'gamma':0.6,'mu':0.9,'layers':[51]}

