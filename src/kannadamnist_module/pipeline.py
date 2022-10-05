import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import config
from my_utils import my_utils as mu

# Gets the GPU if there is one, otherwise the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device type: ", device)




### DATA LOADING

train_set = mu.load_dataset(config.TRAIN_FILE)
test_images = mu.load_dataset(config.TEST_FILE)
val_set = mu.load_dataset(config.DIG_FILE)
train_set = pd.concat([train_set, val_set], axis=0)

train_images, val_images, train_labels, val_labels = train_test_split(train_set.iloc[:, 1:], 
                                                                     train_set.iloc[:, 0], 
                                                                     test_size=0.1)

# Reset index values
train_images.reset_index(drop=True, inplace=True)
val_images.reset_index(drop=True, inplace=True)
train_labels.reset_index(drop=True, inplace=True)
val_labels.reset_index(drop=True, inplace=True)

# Remove the ID column
test_images = test_images.iloc[:, 1:]

# Plot a sample image
mu.plot_image(train_labels[0], train_images.iloc[[0], :], config.OUTPUT)



## DATA AUGMENTATION

train_trans = transforms.Compose(([
    transforms.ToPILImage(),
    transforms.RandomCrop(config.IMAGE_SIZE),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
    transforms.ToTensor(),
]))

val_trans = transforms.Compose(([
    transforms.ToPILImage(),
    transforms.ToTensor(),
]))

# Custom class for the new Kannada MNIST dataset on Kaggle
# Images come in a csv, not as actual images
# Training set is split before given to this class

class KannadaDataSet(torch.utils.data.Dataset):
    # images df, labels df, transforms
    # uses labels to determine if it needs to return X & y or just X in __getitem__
    def __init__(self, images, labels, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
                    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        data = self.X.iloc[i, :] # gets the row
        # reshape the row into the image size 
        # (numpy arrays have the color channels dim last)
        data = np.array(data).astype(np.uint8).reshape(28, 28, 1) 
        
        # perform transforms if there are any
        if self.transforms:
            data = self.transforms(data)
        
        # if !test_set return the label as well, otherwise don't
        if self.y is not None: # train/val
            return (data, self.y[i])
        else: # test
            return data