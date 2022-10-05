import config
import pandas as pd

# Function to check how many predictions match their true labels
def get_num_correct(predictions, labels):
    return predictions.argmax(dim=1).eq(labels).sum().item()

def load_dataset(file_name):
    _data = pd.read_csv(config.DATAPATH + file_name)
    _data = _data.reset_index(drop=True, inplace=True)
    return _data