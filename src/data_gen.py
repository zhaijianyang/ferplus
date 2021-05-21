import os
import csv
import sys

import cv2
import logging
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class DataManager(object):
    """
    load the dataset ferplus
    """
    def __init__(self, dataset_name='ferplus', dataset_root=None, image_size=(48, 48), data_mode=None, train_mode=None):
        """

        :param dataset_name: select the dataset "CK" or "fer2013"
        :param dataset_path: the dataset location dir
        :param num_classes: the classes number of dataset
        :param image_size: the image size output you want
        :param b_gray_chanel: if or not convert image to gray

        :return the tuple have image datas and image labels
        """
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.data_mode = data_mode
        if dataset_name == 'ferplus':
            self.emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
            self.emotion_count = len(self.emotions)
            self.train_mode = train_mode

            self.get_ferplus()

    def get_ferplus(self):

        self.data = []
        self.labels = []
        self.per_emotion_count = np.zeros(self.emotion_count, dtype=np.int)

        if self.data_mode == 'train':
            folder_path = os.path.join(self.dataset_root, 'FER2013Train')
        elif self.data_mode == 'val':
            folder_path = os.path.join(self.dataset_root, 'FER2013Valid')
        elif self.data_mode == 'test':
            folder_path = os.path.join(self.dataset_root, 'FER2013Test')
        else:
            raise Exception('Incorrect mode, please input train, val or test')

        label_path = os.path.join(folder_path, 'label.csv')

        logging.info("Loading %s" % (os.path.join(folder_path)))
        with open(label_path) as csvfile: 
            emotion_label = csv.reader(csvfile) 
            for row in emotion_label: 

                emotion_raw = list(map(float, row[2:len(row)]))
                emotion = self._process_data(emotion_raw, self.train_mode) 
                idx = np.argmax(emotion)
                if idx < self.emotion_count: # not unknown or non-face 
                    emotion = emotion[:-2]
                    emotion = [float(i)/sum(emotion) for i in emotion]

                    label = self._process_target(emotion, self.train_mode)

                    # load the image
                    image_path = os.path.join(folder_path, row[0])
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, self.image_size)

                    self.data.append(img.astype('float32'))
                    self.labels.append(label)
                    self.per_emotion_count[idx] += 1

        self.data = np.expand_dims(np.asarray(self.data),-1) / 255.0 
        # data = data * 2 - 1
        self.labels = np.array(self.labels, 'float32')

        return self.data, self.labels

    def _process_data(self, emotion_raw, mode):
        '''
        Based on https://arxiv.org/abs/1608.01041, we process the data differently depend on the training mode:

        Majority: return the emotion that has the majority vote, or unknown if the count is too little.
        Probability or Crossentropty: convert the count into probability distribution.abs
        Multi-target: treat all emotion with 30% or more votes as equal.
        '''        
        size = len(emotion_raw)
        emotion_unknown     = [0.0] * size
        emotion_unknown[-2] = 1.0

        # remove emotions with a single vote (outlier removal) 
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size 

        if mode == 'majority': 
            # find the peak value of the emo_raw list 
            maxval = max(emotion_raw) 
            if maxval > 0.5*sum_list: 
                emotion[np.argmax(emotion_raw)] = maxval 
            else: 
                emotion = emotion_unknown   # force setting as unknown 
        elif (mode == 'probability') or (mode == 'crossentropy'):
            sum_part = 0
            count = 0
            valid_emotion = True
            while sum_part < 0.75*sum_list and count < 3 and valid_emotion:
                maxval = max(emotion_raw) 
                for i in range(size): 
                    if emotion_raw[i] == maxval: 
                        emotion[i] = maxval
                        emotion_raw[i] = 0
                        sum_part += emotion[i]
                        count += 1
                        if i >= 8:  # unknown or non-face share same number of max votes 
                            valid_emotion = False
                            if sum(emotion) > maxval:   # there have been other emotions ahead of unknown or non-face
                                emotion[i] = 0
                                count -= 1
                            break
            if sum(emotion) <= 0.5*sum_list or count > 3: # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
                emotion = emotion_unknown   # force setting as unknown 
        elif mode == 'multi_target':
            threshold = 0.3
            for i in range(size): 
                if emotion_raw[i] >= threshold*sum_list: 
                    emotion[i] = emotion_raw[i] 
            if sum(emotion) <= 0.5 * sum_list: # less than 50% of the votes are integrated, we discard this example 
                emotion = emotion_unknown   # set as unknown 
                                
        return [float(i)/sum(emotion) for i in emotion]

    def _process_target(self, target, training_mode):
        '''
        Based on https://arxiv.org/abs/1608.01041 the target depend on the training mode.

        Majority or crossentropy: return the probability distribution generated by "_process_data"
        Probability: pick one emotion based on the probability distribtuion.
        Multi-target: 
        '''
        if training_mode == 'majority' or training_mode == 'crossentropy': 
            return target
        elif training_mode == 'probability': 
            idx             = np.random.choice(len(target), p=target) 
            new_target      = np.zeros_like(target)
            new_target[idx] = 1.0
            return new_target
        elif training_mode == 'multi_target': 
            new_target = np.array(target) 
            new_target[new_target>0] = 1.0
            epsilon = 0.001     # add small epsilon in order to avoid ill-conditioned computation
            return (1-epsilon)*new_target + epsilon*np.ones_like(target)


if __name__ == '__main__':
    train_loader = DataManager(dataset_root='D:/maxipy/emotion_class/FERPlus/data', data_mode='train', train_mode='majority')