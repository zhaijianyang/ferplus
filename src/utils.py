#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

import sys
import os
import csv
import numpy as np
import logging
import random as rnd

def cost_func(training_mode):
    '''
    We use cross entropy in most mode, except for the multi-label mode, which require treating
    multiple labels exactly the same.
    '''
    train_loss = None
    if training_mode == 'majority' or training_mode == 'probability' or training_mode == 'crossentropy': 
        # Cross Entropy.
        train_loss = 'categorical_crossentropy'
    elif training_mode == 'multi_target':
        train_loss = 'binary_crossentropy'

    return train_loss

def display_summary(train_data_reader, val_data_reader, test_data_reader):
    '''
    Summarize the data in a tabular format.
    '''
    emotion_count = train_data_reader.emotion_count
    emotin_header = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

    logging.info("{0}\t{1}\t{2}\t{3}".format("".ljust(10), "Train", "Val", "Test"))
    for index in range(emotion_count):
        logging.info("{0}\t{1}\t{2}\t{3}".format(emotin_header[index].ljust(10), 
                     train_data_reader.per_emotion_count[index], 
                     val_data_reader.per_emotion_count[index], 
                     test_data_reader.per_emotion_count[index]))
        