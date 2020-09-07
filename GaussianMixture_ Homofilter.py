#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:42:33 2020

@author: Elaine
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:41:40 2020
Apply GMM on Holomophic filtered images
@author: Elaine
"""
import cv2
from sklearn.mixture import GaussianMixture as GMM
#Read the image jpeg file and store it using imread() function
#from yellow_color import Yellow_Color

from pathlib import Path
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from homofilt import HomomorphicFilter

WIDTH = 512
HEIGHT = 512
path= Path("../data/all data")

filenames = [str(file) for file in path.iterdir()]

num_files = len(filenames)

orig_img = [np.array(Image.open(file).convert("RGB").resize((WIDTH,HEIGHT))) for file in filenames]
orig_img = np.array(orig_img)/255


#open the homomorphic_transformed

pl = Path("../data/Homomorphic_Transformed")

filenames = [str(file) for file in pl.iterdir()]

num_files = len(filenames)

ims = [np.array(Image.open(file).convert("RGB").resize((WIDTH,HEIGHT))) for file in filenames]
ims = np.array(ims)/255
#concatatenate all the images
pixel_list = [im for im in ims]
pixels = np.concatenate(pixel_list, axis=0)
pixels_reshape = pixels.reshape(-1,3)

#gaussian mixture models
gmm_model = GMM(n_components = 12, covariance_type = 'tied').fit(pixels_reshape)
colors = gmm_model.means_
gmm_labels = gmm_model.predict(pixels_reshape)
segmented_img_shape = gmm_labels.reshape(pixels.shape[0],pixels.shape[1])
segmented_img_shape = 255*segmented_img_shape
plt.imshow(segmented_img_shape)


#learning the labels of yellow clusters belong to 
img_y = cv2.imread("../data/Homomorphic_Transformed/Homomorphic_Lot3.tif")
img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)
img_data = np.array(img_y)/255
r = np.mean(img_data[205:215, 600:1000, 0])
g = np.mean(img_data[205:215, 600:1000, 1])
b = np.mean(img_data[205:215, 600:1000, 2])
rgb = np.array([r,g,b])
yellow_label = gmm_model.predict(rgb.reshape((1, -1)))


for index in range(num_files):
    
    #apply the gmm prediction on each image
    img_labels = gmm_labels[HEIGHT*WIDTH*index:HEIGHT*WIDTH*(index+1)]
    predicted_img = colors[img_labels].reshape((HEIGHT, WIDTH, 3))
    plt.imshow(predicted_img)
    
    

    filename = filenames[index].split('/')[-1].split('.')[0]
    output_name = "Output/Gaussian_MM_Homomorphic/Transformed_"+filename+".jpg"
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    #cv2.imwrite(output_name,predicted_img)
    plt.savefig(output_name)
    plt.clf()

    #create a yellow mask
    mask = (img_labels == yellow_label).reshape((HEIGHT, WIDTH, 1))
    masked_image = predicted_img*mask
    plt.figure(figsize=(6,6))
    plt.imshow(masked_image)
    masked_image = cv2.convertScaleAbs(masked_image, alpha=(255.0))
    output_name = "Output/Gaussian_MM_Homomorphic/Masked_"+filename+".jpg"
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    #cv2.imwrite(output_name, masked_image)
    plt.savefig(output_name)
    plt.clf()

    #gaussian blur 
    gblur = cv2.GaussianBlur(masked_image, (7,7), 0 )
    plt.figure(figsize=(6,6))
    plt.imshow(gblur)
    #gblur.astype(np.uint8)
    plt.title('Gaussian');
    plt.show()

    minval = 150
    maxval = 250
    canny_bF = cv2.Canny(gblur, minval, maxval)
    canny_bF = cv2.resize(canny_bF, (HEIGHT,WIDTH))
    #candy line detection
    plt.figure(figsize=(6,6))
    plt.imshow(canny_bF, cmap = 'gray')
    #plt.title('After Applying Canny Edge Detection');
    plt.show()




    contours, hierarchy = cv2.findContours(canny_bF, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    original_img = ims[index]
    RGBimg = original_img
    img_resized = cv2.resize(RGBimg, (HEIGHT,WIDTH))
    for i in range(len(contours)):
        cv2.drawContours(img_resized, contours[i], -1, (255, 0, 0), 2)
    plt.imshow(original_img)
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])
        box = np.int0(cv2.boxPoints(rect))
        im = cv2.drawContours(img_resized, [box], -1, (255, 0, 0), 2)
    plt.figure(figsize=(6,6))
    plt.imshow(img_resized)
    #plt.title('Image with yellow line detected');
    #change to bgr
    #img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    output_name = "Output/Gaussian_MM_Homomorphic/line_detected_"+filename+".jpg"
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    #cv2.imwrite(output_name, img_resized)
    plt.savefig(output_name)
    plt.clf()


 