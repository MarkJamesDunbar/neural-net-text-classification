# Defining our Neural Network structure
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Image starts as a matrix of size (1, 28, 28)
        
        # Convolutions and batch normalisations
        # Batch norm reduces internal covariate shift
        # Normalises the input feature (subtract batch mean, divide by batch sd)
        # This speeds up neural network training times  
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2) # conv1
        self.conv1_bn = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2) # conv2
        self.conv2_bn = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size=3, stride=1, padding=1) # conv3
        self.conv3_bn = nn.BatchNorm2d(num_features=128)
        
        # Fully connected linear layers and batch normalisations
        self.fc1 = nn.Linear(in_features=128*6*6, out_features=1024) # linear 1
        self.fc1_bn = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512) # linear 2
        self.fc2_bn = nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256) # linear 3
        self.fc3_bn = nn.BatchNorm1d(num_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=64) # linear 4
        self.fc4_bn = nn.BatchNorm1d(num_features=64)
        self.out = nn.Linear(in_features=64, out_features=10) # output
    
    def forward(self, z):
        # Apply Relu then Max Pool function between each convolution layer
        # The size of the image after pooling can be obtained by: 
        # image_size - filter_size + 2*padding / stride_length) + 1
        z = F.relu(self.conv1_bn(self.conv1(z)))
        z = F.max_pool2d(z, kernel_size=2, stride=2)
        z = F.relu(self.conv2_bn(self.conv2(z)))
        z = F.max_pool2d(z, kernel_size=2, stride=2)
        z = F.relu(self.conv3_bn(self.conv3(z)))
        z = F.max_pool2d(z, kernel_size=2, stride=1)

        # Apply Relu function between each fully connected layer
        z = F.relu(self.fc1_bn(self.fc1(z.reshape(-1, 128*6*6))))
        z = F.relu(self.fc2_bn(self.fc2(z)))
        z = F.relu(self.fc3_bn(self.fc3(z)))
        z = F.relu(self.fc4_bn(self.fc4(z)))
        z = self.out(z)
        
        return z