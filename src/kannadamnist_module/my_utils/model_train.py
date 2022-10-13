import torch.nn.functional as F
from . import model_eval as mle

def train_model(network, device, optimizer, scheduler, training_data, batch_size):
    """Trains the model using the training data"""

    epoch_loss = 0
    epoch_correct = 0
    network.train() # training mode
    
    for images, labels in training_data:
        # Put data on device
        X, y = images.to(device), labels.to(device)
        # Obtain predictions
        y_ = network(X)
        
        # Zero the gradients
        optimizer.zero_grad()
        # Calculate the loss
        loss = F.cross_entropy(y_, y)
        # Compute the gradients
        loss.backward()
        # Update weights
        optimizer.step()
        
        epoch_loss += loss.item() * batch_size
        epoch_correct += mle.get_num_correct(y_, y)    
    
    print("Train Loss: ", epoch_loss/batch_size)
    print("Train Acc:  ", epoch_correct/len(training_data))
    
    scheduler.step()
    return optimizer.param_groups[0]["lr"], epoch_loss/batch_size, epoch_correct/len(training_data)
