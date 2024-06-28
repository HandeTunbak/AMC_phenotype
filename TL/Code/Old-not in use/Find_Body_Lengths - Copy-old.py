# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Dr Hande Tunbak)
"""

import os
import sys
import glob
import cv2
import numpy as np
import scipy
import imageio
import matplotlib.pyplot as plt
import pandas as pd
import math
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "svg"

from skimage import io, color, data, filters, measure, morphology, img_as_float, exposure, restoration
from skimage.color import rgb2gray
from skimage.filters import unsharp_mask, threshold_multiotsu, threshold_isodata, threshold_otsu, threshold_minimum, threshold_yen
from skimage.filters.rank import autolevel, enhance_contrast
from skimage.morphology import disk, white_tophat, binary_dilation, remove_small_objects, label


#%%


# Load image
img = cv2.imread('C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_TL/Testing_area/lengths/Left_3x_600.tif')

## Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## Mask of green (36,25,25) ~ (86, 255,255)
# mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
threshold = cv2.inRange(hsv, (40, 25, 25), (100, 255,255))
plt.imshow(threshold)


gray_img = color.rgb2gray(img[:,:,:3])
# plot image
fig, ax = plt.subplots()
plt.imshow(gray_img, cmap='gray')
plt.title('gray_img')
plt.show()

eye_mask = gray_img < threshold
plt.imshow(eye_mask)


# remove small particles from image
cleaned_img_eye_1 = remove_small_objects(eye_mask, min_size=110, connectivity=4)

# plot image
fig, ax = plt.subplots()
plt.imshow(cleaned_img_eye_1, cmap='gray')
plt.title('cleaned_eye_mask')
plt.show()



labels = measure.label(cleaned_img_eye_1)

# set image to trace over
fig = px.imshow(cleaned_img_eye_1, binary_string=True)
fig.update_traces(hoverinfo='skip') # hover is only for label info
# fig.show() # not needed

# set properties for drawing     
props = measure.regionprops(labels, gray_img)
properties = ['area', 'eccentricity', 'perimeter']

# For each label, add a filled scatter trace for its contour,
# and display the properties of the label in the hover of this trace.
for index in range(0, labels.max()):
    label_i = props[index].label
    contour = measure.find_contours(labels == label_i, 0.5)[0]
    y, x = contour.T
    hoverinfo = ''
    for prop_name in properties:
        hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
    fig.add_trace(go.Scatter(
        x=x, y=y, name=label_i,
        mode='lines', fill='toself', showlegend=False,
        hovertemplate=hoverinfo, hoveron='points+fills'))
fig.show()


props_2 = measure.regionprops_table(labels, properties=( 'centroid',
                                                          'area',
                                                          'perimeter',
                                                          'orientation',
                                                          'major_axis_length',
                                                          'minor_axis_length',
                                                          'label',
                                                          'eccentricity'            ) )
x_cordinates= pd.DataFrame([props_2['centroid-1']]).T
x_cordinates.columns=['x_cordinates']

y_cordinates= pd.DataFrame([props_2['centroid-0']]).T
y_cordinates.columns=['y_cordinates']

centroids_table= pd.concat([x_cordinates, y_cordinates ], axis=1)

centroids_table2= centroids_table.sort_values('x_cordinates', ascending=True)

for x ,y in zip(centroids_table2['x_cordinates'], centroids_table2['y_cordinates']):
    print(x)
    print(y)
    
for x, y in enumerate(centroids_table2):
    print (y)
    











