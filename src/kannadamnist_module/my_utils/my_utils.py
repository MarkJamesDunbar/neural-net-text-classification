import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###################
# Data prep utils #
###################

def load_dataset(file_name, datapath):
    """Read input CSV file for a given file name"""
    data = pd.read_csv(datapath + file_name)
    return data

def plot_image(label, image, saveloc):
    """Plot an example image and class from the dataset, and save it to file"""
    fig1 = plt.figure(figsize=(6,6));
    fig1.tight_layout()
    plt.title(f"Class: {label}")
    plt.imshow(image.to_numpy().astype(np.uint8).reshape(28, 28), cmap='gray')
    plt.savefig(saveloc+"sample.png")
    
class KannadaDataSet(torch.utils.data.Dataset):
    """Class for handling input CSVs as images using PIL; also handles specified image augmentation"""
    # Get the passed image vector, label vector and transform config
    def __init__(self, images, image_size, labels, transforms=None):
        self.X = images
        self.image_size = image_size
        self.y = labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        # Get a single image vector
        data = self.X.iloc[i, :] 
        # Reshape the vector into an image of shape 28*28
        data = np.array(data).astype(np.uint8).reshape(self.image_size, self.image_size, 1) 
        
        # Produce any transforms that are provided
        if self.transforms:
            data = self.transforms(data)
        
        # If the data is a training set, provide the label; otherwise do not
        if self.y is not None: # train/val
            return (data, self.y[i])
        else:
            return data


##################
# Training utils #
##################

class Accumulator:  
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):  #y_hat is a matrix; 2nd dimension stores prediction scores for each class.
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) # Predicted class is the index of max score         
    cmp = (y_hat.type(y.dtype) == y)  # because`==` is sensitive to data types
    return float(torch.sum(cmp)) # Taking the sum yields the number of correct predictions.

def evaluate_accuracy(net, data_iter): 
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for _, (X, y) in enumerate(data_iter):
        metric.add(accuracy(net(X.to(torch.device('cuda'))), y.to(torch.device('cuda'))), y.numel())
    return metric[0] / metric[1]

def get_num_correct(predictions, labels):
    """Compares model predictions with actual labels, returns the number of matches"""
    return predictions.argmax(dim=1).eq(labels).sum().item()