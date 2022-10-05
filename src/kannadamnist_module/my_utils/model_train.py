import torch
import torch.nn.functional as F

def get_num_correct(predictions, labels):
    """Compares model predictions with actual labels, returns the number of matches"""
    return predictions.argmax(dim=1).eq(labels).sum().item()



def train_model(network, device, optimizer, scheduler, training_data, batch_size):
    """Trains the model using the training data"""
    epoch_loss = 0
    epoch_correct = 0
    network.train() # training mode
    
    for images, labels in training_data:
        X, y = images.to(device), labels.to(device) # put X & y on device
        y_ = network(X) # get predictions
        
        optimizer.zero_grad() # zeros out the gradients
        loss = F.cross_entropy(y_, y) # computes the loss
        loss.backward() # computes the gradients
        optimizer.step() # updates weights
        
        epoch_loss += loss.item() * batch_size
        epoch_correct += get_num_correct(y_, y)    

    scheduler.step()
    return optimizer.param_groups[0]["lr"]


def eval_model(network, device, validation_data, validation_set_size, batch_size):
    # Evaluation with the validation set
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