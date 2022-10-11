#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Filepath Config
DATAPATH="datasets"
OUTPUT="output"
TRAIN_FILE="train.csv"
TEST_FILE="test.csv"
SAMPLE_FILE ="sample_submission.csv"
DIG_FILE="Dig-MNIST.csv"

# Data Config
SPLITSIZE = 0.2
NUM_CLASSES = 10
CLASS_NAMES = {0:"Omdu", 1:"Eradu", 2:"Muru", 3:"Nalku", 4:"Aidu", 5:"Aru", 6:"Elu", 7:"Emtu", 8:"Ombattu", 9:"Hattu"}
IMAGE_SIZE = 28

# Model Config
INITIAL_LR = 0.01
BATCH_SIZE = 100
EPOCHS = 50

