#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:13:41 2020

@author: Elaine
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:58:31 2020

@author: Elaine
"""
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture as GMM
#Read the image jpeg file and store it using imread() function
#from yellow_color import Yellow_Color

from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
#from yellow_color import Yellow
import os
from homofilt import HomomorphicFilter

WIDTH = 512
HEIGHT = 512

pl = Path("../data/all data")
#rd = Path("Road")

filenames = [str(file) for file in pl.iterdir()]
#filenames = filenames + [str(file) for file in rd.iterdir()]

num_files = len(filenames)

ims = [np.array(Image.open(file).convert("RGB")) for file in filenames]
ims = np.array(ims)
for index in range(len(ims)):
    original_img = ims[index]
    #original_img = cv2.resize(original_img, (WIDTH, HEIGHT))
    hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
    # Main code
    img = hsv[:,:,2]
    homo_filter = HomomorphicFilter(a = 0.75, b = 1.25)
    img_filtered = homo_filter.filter(I=img, filter_params=[30,2])
    plt.imshow(img_filtered)
    d = np.mean(img)- np.mean(img_filtered)
    img_filtered = img_filtered + d
    hsv[:,:,2] = img_filtered
    filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    plt.imshow(filtered)
    filename = filenames[index].split('/')[-1].split('.')[0]
    output_name = "../data/Holomorphic_Transformed/Homomorphic_"+filename+".tif"
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
   # plt.savefig(output_name)
    plt.imshow(filtered)
    cv2.imwrite(output_name, filtered)
    plt.clf()


