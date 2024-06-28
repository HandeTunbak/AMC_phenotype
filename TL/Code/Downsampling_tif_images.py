# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:30:46 2024

@author: hande
"""

import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image, ImageEnhance, ImageOps
import cv2

    


folder= 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_TL/Testing_area/Injected/Fish4a/Original - Copy' 


img = cv2.imread(folder + '/Right_3x_064.tif')


# img = cv2.imread(folder + '/Right_3x_101.tif')


downscale_factor = 1.62


downsampled_img = cv2.resize(img, None, fx=1/downscale_factor, fy=1/downscale_factor, interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite((folder + '/Downsampled-Right_3x_04.tif'), downsampled_img)



