import torch
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split

from config import config

from my_utils import data_utils as du
from my_utils import model_train as mlt
from my_utils import model_eval as mle
from model import model


# Use the GPU if available, otherwise select the cpu (default)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device type: ", device)

##############################
# DATA LOADING/PREPROCESSING #
##############################

train_set = du.load_dataset(config.TRAIN_FILE, config.DATAPATH)
test_images = du.load_dataset(config.TEST_FILE, config.DATAPATH)
val_set = du.load_dataset(config.DIG_FILE, config.DATAPATH)
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
du.plot_image(train_labels[0], config.CLASS_NAMES, train_images.iloc[[0], :], config.OUTPUT)

# Define transform steps to convert csv data to images, and augment training data
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
train_set = du.KannadaDataSet(train_images, config.IMAGE_SIZE, train_labels, train_trans)
val_set = du.KannadaDataSet(val_images, config.IMAGE_SIZE, val_labels, val_trans)
test_set = du.KannadaDataSet(test_images, config.IMAGE_SIZE, None, val_trans)

##################
# Model Training #
##################

# Create network on available device
net = model.Network().to(device)

# Acquire a summary of the number of parameters in the Neural Network
params = mle.get_n_params(net)
print(f"There are {params} total parameters in the neural network.")

# Create dataset dataloaders
training_dataloader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)

# Define our optimiser: we're using Adam
optimizer = optim.Adam(net.parameters(), lr=config.INITIAL_LR)
# optimizer = optim.Adagrad(net.parameters(), lr=INITIAL_LR)
# optimizer = optim.SGD(net.parameters(), lr=INITIAL_LR, nesterov=True, momentum=0.9)

# Define a Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

# Empty lists for appending loss/acc values for plotting later
lrs = []
train_loss = []
train_acc = []
val_loss = []
val_acc = []

# Run our epoch cycles
for epoch in range(config.EPOCHS):
    # Print epoch cycle
    print("===============================================")
    print(f"Epoch Cycle: {epoch+1}")
    print("-----------------------------------------------")
    # Train the model, and append the current learning rate
    lr, tl, ta = mlt.train_model(net, device, optimizer, scheduler, training_dataloader, config.BATCH_SIZE)
    # Append the values to our lists for plotting
    lrs.append(lr)
    train_loss.append(tl)
    train_acc.append(ta)
    # Evaluate the model, return the validation loss and validation accuracy
    vl, va = mle.eval_model(net, device, validation_dataloader, len(val_images), config.BATCH_SIZE)
    # Append the values to our lists for plotting
    val_loss.append(vl)
    val_acc.append(va)

####################################
# Model Evaluation Plot Generation #
####################################

# Make sure the model is in the correct mode (eval)
net.eval()

# Create empty tensor for predictions
predictions = torch.LongTensor().to(device)

# Use trained model to generate predictions
for images, _ in validation_dataloader:
    preds = net(images.to(device))
    predictions = torch.cat((predictions, preds.argmax(dim=1)), dim=0)

# Plot confusion matrix
mle.confusion_matrix(config.NUM_CLASSES, val_labels, predictions, config.OUTPUT)

# Plot the model's training/validation loss curves
mle.train_loss_curve(config.EPOCHS, train_loss, val_loss, config.OUTPUT)

# Plot model's training/validation accuracy curves
mle.train_acc_curve(config.EPOCHS, train_acc, val_acc, config.OUTPUT)

####################################
# Model Submission File Generation #
####################################

# Add the test set to a DataLoader
test_dl = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)

# Make sure the model is in eval mode!
net.eval()

# Create a tensor for predictions
predictions = torch.LongTensor().to(device) 

# Save the new predictions
for images in test_dl:
    preds = net(images.to(device))
    predictions = torch.cat((predictions, preds.argmax(dim=1)), dim=0)

# Edit the existing sample submission file with our own predictions
submission_file = du.load_dataset(config.SAMPLE_FILE, config.DATAPATH)

# Rewrite the label column with our predictions
# As predictions is a tensor on the CPU, covert it to a numpy array to be saved
submission_file['label'] = predictions.cpu().numpy()

# Save the dataframe as a new submission csv!
submission_file.to_csv(config.OUTPUT + "submission/submission.csv", index=False)