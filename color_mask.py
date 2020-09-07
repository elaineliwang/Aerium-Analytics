#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:08:45 2020

@author: Elaine
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

class color_mask:

    def __init__(self, filename):
        #self.joblib_file = "color_mask/"+filename.split('/')[-1].split('.')[0]+"_"+str(K)+".pkl"

        self.img = cv2.imread(filename)
    
    def fit(self):
            self.img = cv2.imread(filename)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
    def output_mask(self):
        # define range of yellow color in HSV
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([30,255,255])

        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(self.img, lower_yellow, upper_yellow)

        # Bitwise-AND mask and original image
        output = cv2.bitwise_and(self.img, self.img, mask = mask)
        ratio = cv2.countNonZero(mask)/(self.img.size/3)
        print('pixel percentage:', np.round(ratio*100, 2))
        plt.imshow(mask)
        plt.show()