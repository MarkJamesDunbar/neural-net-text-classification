# Neural-Network-for-Kannada-MNIST-Classification
 Kannada MNIST Neural Network Classifier

## Setup
### Virtual Environment
Create a virtual environment within the inside the `Neural-Network-for-Kannada-MNIST-Classification` directory before installing dependencies.

#### Windows
To create the virtual environment, run:
`python -m venv <venv name>`

To activate the virtual environment, run:
`<venv name>/Scripts/activate.bat`

And to use an existing virtual environment, run:
`source <venv name>/Scripts/activate`

#### Mac OS
Create the virtual environment by running
`python -m venv <venv name>`

To activate the virtual environment, run:
`<venv name>/bin/activate`

And to use an existing virtual environment, run:
`source venv/bin/activate`

### Install requirements
To set up the requirements for this python package, please run:
`python setup.py install`
or
`pip install -e .`
inside the `Neural-Network-for-Kannada-MNIST-Classification` directory

## Running the model training and validation pipeline
To run the pipeline, use the following command from the `Neural-Network-for-Kannada-MNIST-Classification` directory:
`python src/kannadamnist_module/pipeline.py`

## Output files
The model will output several files in the directory `/src/kannadamnist_module/output`. Here there are 3 directories:
 
* `sample_data` - a sample image and related class label for one of the kannada character images in the dataset

* `model_evaluation` - 4 plots:
    * The model's confusion matrix on the validation data
    * The model's loss curve on the training and validation data over epochs
    * The model's accuracy curve on the trainind and validation data over all epochs
    * The model's change in learning rate over all epochs

* `submission_file` - a csv containing the image ID and model prediction for the kaggle test data. This is the file submitted to kaggle to obtain the kaggle score for this model.

## Kaggle submission
The kaggle notebook used to test and produce this CNN python package can be found at:
https://www.kaggle.com/code/markdunbar/kannadamnist-cnn/edit/run/107590957

The kaggle score for this model is shown below:
![image](https://user-images.githubusercontent.com/57494763/195201716-44566628-6f87-4f95-b214-395703c1aa7d.png)
