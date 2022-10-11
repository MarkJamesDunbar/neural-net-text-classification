import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


def get_num_correct(predictions, labels):
    """Compares model predictions with actual labels, returns the number of matches"""

    return predictions.argmax(dim=1).eq(labels).sum().item()


def eval_model(network, device, validation_data, validation_set_size, batch_size):
    """Evaluation with the validation set"""

    # Ensure the model is in eval mode (this disables dropout/batchnorm etc.)
    network.eval() 
    val_loss = 0
    val_correct = 0
    # Set all requires_grad() flags to false
    with torch.no_grad():
        # Loop through our validation data, generate predictions and add to the the loss/accuracy count for each image
        for images, labels in validation_data:
            # Put X and Y on the device
            X, y = images.to(device), labels.to(device) # to device
            # Obtain the predictions
            preds = network(X) 
            # Calculate the cross entropy loss
            loss = F.cross_entropy(preds, y)
            val_correct += get_num_correct(preds, y)
            val_loss = loss.item() * batch_size
    # Print the loss and accuracy for the validation set
    print("Validation Loss: ", val_loss)
    print("Validation Acc:  ", (val_correct/validation_set_size)*100)
    # Return loss and accuracy values
    return val_loss, ((val_correct/validation_set_size)*100)

def get_n_params(model):
    """Summarizes the number of parameters in the neural network"""
    
    params_counter = 0
    for param in list(model.parameters()):
        nn=1
        for s in list(param.size()):
            nn = nn*s
        params_counter += nn
    return params_counter

def confusion_matrix(num_classes, validation_labels, predictions, datapath):
    """Generate and save a confusion matrix for the Neural Network"""

    # Generate the confusion matrix
    cmt = torch.zeros(num_classes, num_classes, dtype=torch.int32)
    for i in range(len(validation_labels)):
        cmt[validation_labels[i], predictions[i]] += 1
    # Detatch from CPU and convert to numpy array 
    cmt = cmt.cpu().detach().numpy()
    df_cm = pd.DataFrame(cmt/np.sum(cmt) *10, index = [i for i in range(num_classes)],
                        columns = [i for i in range(num_classes)])
    # Plot the dataframe using seaborn
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True)
    plt.title("Neural Network Confusion Matrix", fontsize=20)
    plt.savefig(os.path.join(datapath,"model_evaluation","confusion_matrix"))

def learning_rate_curve(epochs, lrs, datapath):
    """Generate a plot of the learning rate with each epoch"""

    # Create list of epochs for plotting purposes
    epoch_num = list(range(1, epochs+1))
    # Plot the curves
    plt.figure(figsize=(12,7))
    plt.title("Neural Network Learning Rate Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate Value")
    plt.grid()
    plt.plot(epoch_num, lrs, 'g-')
    plt.xticks(epoch_num)
    plt.savefig(os.path.join(datapath,"model_evaluation","learning_rate_curve"))

def loss_curves(epochs, train_loss, val_loss, datapath):
    """Generate and save a training loss/acc curve for the model's training"""

    # Create list of epochs for plotting purposes
    epoch_num = list(range(1, epochs+1))
    # Plot the curves
    plt.figure(figsize=(12,7))
    plt.title("Neural Network Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.grid()
    plt.plot(epoch_num, train_loss, 'b-', label='training loss')
    plt.plot(epoch_num, val_loss, 'r-', label='validation loss')
    plt.legend(loc="upper right")
    plt.xticks(epoch_num)
    plt.savefig(os.path.join(datapath,"model_evaluation","loss_curves"))


def acc_curves(epochs, train_acc, val_acc, datapath):
    """Generate and save a training accuracy curve for the model's training"""

    # Create list of epochs for plotting purposes
    epoch_num = list(range(1, epochs+1))
    # Plot the curves
    plt.figure(figsize=(12,7))
    plt.title("Neural Network Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(97,100)
    plt.grid()
    plt.plot(epoch_num, train_acc, 'b-', label='training accuracy')
    plt.plot(epoch_num, val_acc, 'r-', label='validation accuracy')
    plt.legend(loc="lower right")
    plt.xticks(epoch_num)
    plt.savefig(os.path.join(datapath,"model_evaluation","acc_curves"))
