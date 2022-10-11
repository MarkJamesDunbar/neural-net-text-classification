import torch.nn.functional as F
from model_eval import get_num_correct

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
    return optimizer.param_groups[0]["lr"], epoch_loss, epoch_correct/len(training_data)
