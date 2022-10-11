import os
import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###################
# Data prep utils #
###################

def load_dataset(file_name, datapath):
    """Read input CSV file for a given file name"""
    data = pd.read_csv(os.path.join(datapath,file_name))
    return data

def plot_image(label, classnames, image, datapath):
    """Plot an example image and class from the dataset, and save it to file"""
    fig1 = plt.figure(figsize=(6,6));
    fig1.tight_layout()
    plt.title(f"Class: {label}, Name: {classnames[label]}")
    plt.imshow(image.to_numpy().astype(np.uint8).reshape(28, 28), cmap='gray')
    plt.savefig(os.path.join(datapath,"sample_data","sample.png"))
    
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


