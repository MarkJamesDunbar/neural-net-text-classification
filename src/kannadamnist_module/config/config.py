#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Filepath Config
DATAPATH="./datasets/"
OUTPUT="output/"
TRAIN_FILE="train.csv"
TEST_FILE="test.csv"
SAMPLE_FILE ="sample_submission.csv"
DIG_FILE="Dig-MNIST.csv"


# Data Config
SPLITSIZE = 0.2
NUM_CLASSES = 10
CLASS_NAMES = {0:"omdu", 1:"eradu", 2:"muru", 3:"nalku", 4:"aidu", 5:"aru", 6:"elu", 7:"emtu", 8:"ombattu", 9:"hattu"}
IMAGE_SIZE = 28


# Model Config
INITIAL_LR = 0.01
BATCH_SIZE = 100
EPOCHS = 10

