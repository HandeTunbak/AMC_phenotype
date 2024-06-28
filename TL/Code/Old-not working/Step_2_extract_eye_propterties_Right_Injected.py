# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:33:20 2022

@author: hande
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 12:24:07 2022

@author: handeo
"""

''' This code is to measure the legnths and height of fish not treated in PTU. '''

#%% # Import useful libraries 
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
from pathlib import Path

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "svg"

import skimage
from skimage import io, color, data, filters, measure, morphology, img_as_float, exposure, restoration
from skimage.color import rgb2gray
from skimage.filters import unsharp_mask, threshold_multiotsu, threshold_isodata, threshold_otsu, threshold_minimum, threshold_yen
from skimage.filters.rank import autolevel, enhance_contrast
from skimage.morphology import disk, white_tophat, binary_dilation, remove_small_objects, label

#%% # Import useful local libraries 
# define path of local library
lib_path = r'C:/Repos/Handes_local_repos/Wilson_Local_Python_Libs'
# add local lib path to system paths for use
sys.path.append(lib_path)
# import specific local libs
import Image_manipulation as img_manip

#%% Search through folder/s to find image files 
# Location with subdirectories
# Experiment_folder = 'C:/Users/hande/OneDrive/Desktop/Code_testing/Experiment_1'

Experiment_folder = 'D:/Code_testing/WLS_TL/Testing_area/'
#'F:/Code_testing/Experiment_testing/ZFHX4_experiment/'

# Set experiment condtion to be analysed '
Experiment_condition = '/Injected/**/Cropped/' 
# Experiment_condition = '/Control/**/Cropped/' 


# Get list of all images
# for all jpng image files
# image_paths_all = glob.glob(Experiment_folder + 
#                             Experiment_condition + '/**/*.jpg', recursive=True)
# for all jpng image files called left cropped
img_paths_left = glob.glob(Experiment_folder + 
                              Experiment_condition + '/Left_8x' + '**' +'.tif', recursive=False)

# # for all jpng img files called right cropped
img_paths_right = glob.glob(Experiment_folder + 
                              Experiment_condition + '/Right_8x' + '**' +'.tif', recursive=False)

# Set empty list for later
x_lengths_all = []
y_lengths_all = []
eye_parameters_all = pd.DataFrame([])
i=1
#%% Load images

# # for all images
# for f, fish_images in enumerate(image_paths_all):
#     image = skimage.io.imread(fish_images)

#     fig, ax = plt.subplots()
#     plt.imshow(image)
#     plt.show()
#     filename = 'image_paths_all'

# # ---------------------    
# # # for left side imgs  
# for f, fish_imgs in enumerate(img_paths_left):
#     img = io.imread(fish_imgs)
    
#     # plot image
#     fig, ax = plt.subplots()
#     plt.imshow(img)
#     plt.title('original_image')
#     plt.show()    
#     filename = 'image_paths_left'
        
# #--------------------- 
# # # for right side images       
for f, fish_imgs in enumerate(img_paths_right):
    img = io.imread(fish_imgs)
    
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(img)
    plt.title('original_image')
    plt.show()    
    filename = 'image_paths_right'
    

#%% # manipulate images 

    # crop image to get rid of uneccessary background 
    # img_manip.crop_around_center(img, length, height)    
    # orgininal image size =>  height = 1944 and length = 2592
    cropped_img = img_manip.crop_around_center(img, 2700, 3600)
    
    # convert the image to grayscale
    #gray_img = color.rgb2gray(io.imread(cropped_img)[:,:,:3])
    gray_img = color.rgb2gray(cropped_img[:,:,:3])
    
    
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(gray_img, cmap='gray')
    plt.title('gray_image')
    plt.show()
    
    
    # denoise the image via mdeian filter 
    denoised_img = filters.median(gray_img)
    
    # plot image
    # fig, ax = plt.subplots()
    # plt.imshow(denoised_img, cmap='gray')
    # plt.title('denoised_image')
    # plt.show()
    
    # sharpen image with unsharp mask filter with strel (strucutral element )
    #  The sharp details are identified as a difference between the original image and its blurred version. 
    sharp_img = unsharp_mask(denoised_img, radius=8, amount=2)
    
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(sharp_img, cmap='gray')
    plt.title('sharpened_image')
    plt.show()
    


    # # Rank image intensities 
    # leveled_img = enhance_contrast (sharp_img, disk(10))
    
    # # plot image
    # fig, ax = plt.subplots()
    # plt.imshow(leveled_img, cmap='gray')
    # plt.title('leveled_image')
    # plt.show()

#%%

    # create a histogram of the blurred grayscale image
    histogram, bin_edges = np.histogram(sharp_img, bins=256, range=(0.0, 1.0))
    
    plt.plot(bin_edges[0:-1], histogram)
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim(0, 1.0)
    plt.title('histogram_pixel_intensities')
    plt.show()




 #%%   
#--------  

# Eye 
    ## create a mask based on the threshold
    ## automaitically find best threshold
   
    ## threshold = threshold_isodata(sharp_img, nbins=256)
    eye_threshold = threshold_multiotsu(sharp_img)
    eye_threshold = (eye_threshold[0]/1.5)
    #eye_threshold = (eye_threshold[0] /1.45)
    #eye_threshold = (eye_threshold[0] /0.5)
    
    #eye_threshold = skimage.filters.threshold_otsu(sharp_img)
    #eye_threshold = threshold_minimum(sharp_img)
    #eye_threshold = threshold_yen(sharp_img)
        
    ## If the regions of interests is darker than the background use $ gray_image > t
    ## If the regions interests is lighter than the background use $ gray_image < t
    eye_mask = sharp_img < eye_threshold

    # plot image
    # fig, ax = plt.subplots()
    # plt.imshow(eye_mask, cmap='gray')
    # plt.title('sharp_img')
    # plt.show()
    
#-------- 
 
    # remove small particles from image
    cleaned_img_eye_1 = remove_small_objects(eye_mask, min_size=5500, connectivity=1)
    
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(cleaned_img_eye_1, cmap='gray')
    plt.title('cleaned_eye_mask')
    plt.show()

#--------    
  
    # # dilate image to close gaps
    # footprint = disk(5.5)
    # dilated_eye_mask = binary_dilation(cleaned_img_eye_1, footprint)
        
    # # plot image
    # fig, ax = plt.subplots()
    # plt.imshow(dilated_eye_mask, cmap='gray')
    # plt.title('dilated_eye_mask')
    # plt.show()
   
#--------  
 # Label areas in mask
    # labels = measure.label(dilated_eye_mask)
    labels = measure.label(cleaned_img_eye_1)
    
    # set image to trace over
    fig = px.imshow(cleaned_img_eye_1, binary_string=True)
    fig.update_traces(hoverinfo='skip') # hover is only for label info
    # fig.show() # not needed
    
    # set properties for drawing     
    props = measure.regionprops(labels, cropped_img)
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
        
    
    # =-----------------
       
    ## centroid: array --> Centroid coordinate tuple (row, col).(row, col) y and x axis--> Where row is y values and col is x values.
    ## area: int --> Number of pixels in region
    ## preimeter: float --> Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
    ## orientation: float --> Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.
    ## axis_minor_length: float --> The length of the minor axis of the ellipse that has the same normalized second central moments as the region   
    ## axis_major_length: float --> The length of the major axis of the ellipse that has the same normalized second central moments as the region.    
    ## eccentricity: float --> Eccentricity of the ellipse that has the same second-moments as the region. 
        # The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). 
        # When it is 0, the ellipse becomes a circle.
    props_2 = measure.regionprops_table(labels, properties=( 'centroid',
                                                             'area',
                                                             'perimeter',
                                                             'orientation',
                                                             'major_axis_length',
                                                             'minor_axis_length',
                                                             'label',
                                                             'eccentricity') )
    
    # if fish facing the left, the eye has the smallest X value. 
    if filename == 'image_paths_left':
        # reorder df with smallest X value first
        sorted_df_eye = pd.DataFrame(props_2).sort_values(by='centroid-1', ascending=True)
        sorted_df_eye = sorted_df_eye.reset_index(drop=True)
        eye_properties = sorted_df_eye.loc[0]
        
    # if fish facing the right, the eye has the largest X value.         
    elif filename == 'image_paths_right':
        # reorder df with largest X value first
        sorted_df_eye = pd.DataFrame(props_2).sort_values(by='centroid-1', ascending=False)
        sorted_df_eye = sorted_df_eye.reset_index(drop=True)
        eye_properties = sorted_df_eye.loc[0]
        
    else: 
        print('error on sorting regions')
        
    
    
    regions = measure.regionprops(labels)
    
    fig, ax = plt.subplots()
    ax.imshow(cleaned_img_eye_1, cmap=plt.cm.gray)
    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length
    
        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)
    
        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)
        
    
    print('Fish ' +str(i))
    i=i+1
    eye_parameters_all= pd.concat([eye_parameters_all, eye_properties], axis=1)



#%%   # Summarise findings on one table 

if filename == 'image_paths_left': 
    left_eye_parameters_all = pd.DataFrame(eye_parameters_all).T
    left_eye_parameters_all= left_eye_parameters_all.reset_index(drop=True)
    
    left_eye_parameters_all.to_csv( str(Path(img_paths_left[f]).parents[2]) + '\Left_eye_parameters.csv')
    
elif filename == 'image_paths_right': 
    right_eye_parameters_all = pd.DataFrame(eye_parameters_all).T
    right_eye_parameters_all= right_eye_parameters_all.reset_index(drop=True)
    
    right_eye_parameters_all.to_csv( str(Path(img_paths_right[f]).parents[2]) + '\Right_eye_paramters.csv')
    
else:
    print ('Error saving csv data')   


