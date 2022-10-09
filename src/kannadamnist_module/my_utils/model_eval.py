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
    network.eval() # eval mode
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for images, labels in validation_data:
            X, y = images.to(device), labels.to(device) # to device
            
            preds = network(X) # get predictions
            loss = F.cross_entropy(preds, y) # calculate the loss
            
            val_correct += get_num_correct(preds, y)
            val_loss = loss.item() * batch_size

    # Print the loss and accuracy for the validation set
    print(" Validation Loss: ", val_loss)
    print(" Validation Acc: ", (val_correct/validation_set_size)*100)

    # Return loss and accuracy values
    return val_loss, ((val_correct/validation_set_size)*100)


def confusion_matrix(num_classes, validation_labels, predictions, output_folder):
    """Generate and save a confusion matrix for the Neural Network"""
    # Make the confusion matrix
    cmt = torch.zeros(num_classes, num_classes, dtype=torch.int32)
    for i in range(len(validation_labels)):
        cmt[validation_labels[i], predictions[i]] += 1

    # Convert to numpy array 
    cmt = cmt.cpu().detach().numpy()

    df_cm = pd.DataFrame(cmt/np.sum(cmt) *10, index = [i for i in range(num_classes)],
                        columns = [i for i in range(num_classes)])
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True)
    plt.title("Neural Network Confusion Matrix")
    plt.savefig(output_folder + "model_evaluation/confusion_matrix")

def train_loss_curve(epochs, loss, output_folder):
    """Generate and save a training loss/acc curve for the model's training"""
    epoch_num = list(range(1, epochs+1))

    plt.figure(figsize=(12,7))
    plt.title("Neural Network Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.grid()
    plt.plot(epoch_num, loss, 'b-')
    plt.xticks(epoch_num)
    plt.savefig(output_folder + "model_evaluation/train_loss_curve")

def train_acc_curve(epochs, acc, output_folder):
    """Generate and save a training accuracy curve for the model's training"""
    epoch_num = list(range(1, epochs+1))

    plt.figure(figsize=(12,7))
    plt.title("Neural Network Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Value (%)")
    plt.ylim(97,100)
    plt.grid()
    plt.plot(epoch_num, acc, 'r-')
    plt.xticks(epoch_num)
    plt.savefig(output_folder + "model_evaluation/train_acc_curve")
