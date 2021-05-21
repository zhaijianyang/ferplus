#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

import sys
import time
import os
import math
import csv
import argparse
import numpy as np
import logging

import keras
from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import accuracy_score

from models import mini_XCEPTION
from data_gen import DataManager
from utils import display_summary, cost_func
    
def main():

    # Specify the training mode: majority, probability, crossentropy or multi_target
    training_mode = 'majority'

    # create needed folders.
    log_path   = os.path.join('./log', training_mode)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # creating logging file
    
    logging.basicConfig(filename = os.path.join(log_path, "train_info.log"), filemode = 'w', level = logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info("Starting with training mode {}.".format(training_mode))

    base_folder = './data'
    num_classes = 8
    input_shape = (48, 48, 1) # (64, 64, 1)
    batch_size = 1024
    max_epochs = 100

    logging.info("Loading data...")
    train_loader = DataManager(dataset_root=base_folder, image_size=input_shape[:2], data_mode='train', train_mode=training_mode)
    val_loader = DataManager(dataset_root=base_folder, image_size=input_shape[:2], data_mode='val', train_mode='majority')
    test_loader = DataManager(dataset_root=base_folder, image_size=input_shape[:2], data_mode='test', train_mode='majority')

    display_summary(train_loader, val_loader, test_loader)

    # data generator
    data_generator = ImageDataGenerator(
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=.1,
                            horizontal_flip=True)
    # create the model
    model = mini_XCEPTION(input_shape, num_classes)
    model.summary()

    train_loss = cost_func(training_mode)
    model.compile(optimizer='adam', loss=train_loss, metrics=['accuracy'])
    

    #training the model
    logging.info("Start training...")
    log_file_path = os.path.join(log_path, "train.log")
    csv_logger = CSVLogger(log_file_path, append=False)

    save_model_path = os.path.join(log_path, 'fer_xception.h5')
    callbacks=[
                ModelCheckpoint(save_model_path,
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True),
                csv_logger
            ]

    model.fit(data_generator.flow(train_loader.data[:100], train_loader.labels[:100], batch_size=batch_size),
            epochs=max_epochs,
            steps_per_epoch=100 / batch_size,
            verbose=1,
            validation_data=(val_loader.data, val_loader.labels),
            shuffle=True,
            callbacks=callbacks)

    # test
    logging.info("Start test...")
    model = load_model(save_model_path)
    pred = model.predict(test_loader.data)

    pred = np.argmax(pred, axis=1)
    true = np.argmax(test_loader.labels, axis=1)
    test_accuracy = accuracy_score(true, pred)

    logging.info("Test Accuracy: {}".format(test_accuracy))

    
if __name__ == "__main__":
    main()