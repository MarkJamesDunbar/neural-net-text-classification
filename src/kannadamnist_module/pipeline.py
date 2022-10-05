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
from my_utils import model_train
from model import model


# Use the GPU if available, otherwise select the cpu (default)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device type: ", device)

### DATA LOADING
train_set = mu.load_dataset(config.TRAIN_FILE, config.DATAPATH)
test_images = mu.load_dataset(config.TEST_FILE, config.DATAPATH)
val_set = mu.load_dataset(config.DIG_FILE, config.DATAPATH)
train_set = pd.concat([train_set, val_set], axis=0)

# Split the data into train/test splits - splitsize defined in config
train_images, val_images, train_labels, val_labels = train_test_split(train_set.iloc[:, 1:], 
                                                                     train_set.iloc[:, 0], 
                                                                     test_size=config.SPLITSIZE)

# Reset index values
train_images.reset_index(drop=True, inplace=True)
val_images.reset_index(drop=True, inplace=True)
train_labels.reset_index(drop=True, inplace=True)
val_labels.reset_index(drop=True, inplace=True)

# Remove ID column
test_images = test_images.iloc[:, 1:]

# Plot a sample image, save it to the output folder
mu.plot_image(train_labels[0], config.CLASS_NAMES, train_images.iloc[[0], :], config.OUTPUT)

## Define transform steps to convert csv data to images, and augment training data
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

# Perform our transforms and augmentations
train_set = mu.KannadaDataSet(train_images, config.IMAGE_SIZE, train_labels, train_trans)
val_set = mu.KannadaDataSet(val_images, config.IMAGE_SIZE, val_labels, val_trans)
test_set = mu.KannadaDataSet(test_images, config.IMAGE_SIZE, None, val_trans)

# create network on available device
network = model.Network().to(device)

training_dataloader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)

# Define our optimiser: we're using Adam
optimizer = optim.Adam(network.parameters(), lr=config.INITIAL_LR)

# Adding a Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

lrs = []
val_loss = []
val_acc = []


# Run our Epoch cycles
for epoch in range(config.EPOCHS):
    # Print epoch cycle
    print(f"Epoch Cycle: {epoch+1}")

    # Train the model, and append the current learning rate
    lrs.append(model_train.train_model(network, device, optimizer, scheduler, training_dataloader, config.BATCH_SIZE))


    # Evaluate the model, return the validation loss and validation accuracy
    loss, acc = model_train.eval_model(network, device, validation_dataloader, len(val_images), config.BATCH_SIZE)
    val_loss.append(loss)
    val_acc.append(acc)

print("woooo")
print(lrs)
