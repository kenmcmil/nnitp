#import numpy

#from tensorflow.keras.models import load_model
import torch
from torchvision import datasets, transforms
#from nnitp.models.models import MNIST_Model
#from nnitp.model_wrapper import Wrapper
from models.models import MNIST_Model
from model_wrapper import Wrapper
#from nnitp.datatype import Image

# Fetch the CIFAR10 dataset, normalized with mean=0, std = 1

def get_data():
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,),(0.3081,))
    ])

    train_data = datasets.MNIST('data', train = True, download = True, transform = transform)
    test_data = datasets.MNIST('data', train = False, download = True, transform = transform)
    return train_data, test_data

# Fetch the trained model

def get_model():
    model = MNIST_Model()
    model.load_state_dict(torch.load('mnist_model.pth'))
    return Wrapper(model,[1,1,28,28])

params = {'size':100,'alpha':0.98,'gamma':0.6,'mu':0.9,'layers':[2]}

