# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:33:20 2022

@author: hande
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 12:24:07 2022

@author: hande
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

#%% # Import useful local libraries 
# define path of local library
lib_path = r'C:/Repos/Handes_local_repos/Wilson_Local_Python_Libs'
# add local lib path to system paths for use
sys.path.append(lib_path)
# import specific local libs
import Image_manipulation as img_manip

#%% Set exerpeiment folder 
Experiment_folder = Experiment_folder = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/'

#%%
# # Set flags
# condition = 'Control'
condition = 'Injected' 

direction = 'Left'
# direction = 'Right'

# pheno = 'MAB_siblings'
pheno = 'MAB_phenos'

## Control
if condition == 'Control' and direction == 'Left' and pheno == 'MAB_siblings':
    Experiment_condition = '/Control/MAB_siblings/**/Original - Copy' 

elif condition == 'Control' and direction == 'Right' and pheno == 'MAB_siblings':
    Experiment_condition = '/Control/MAB_siblings/**/Original - Copy' 
    
elif condition == 'Control' and direction == 'Left' and pheno == 'MAB_phenos':
    Experiment_condition = '/Control/MAB_phenos/**/Original - Copy' 

elif condition == 'Control' and direction == 'Right' and pheno == 'MAB_phenos':
    Experiment_condition = '/Control/MAB_phenos/**/Original - Copy' 

## Injected
elif condition == 'Injected' and direction == 'Left' and pheno == 'MAB_siblings':
    Experiment_condition = '/Injected/MAB_siblings/**/Original - Copy'
    
elif condition == 'Injected' and direction == 'Right' and pheno == 'MAB_siblings':
    Experiment_condition = '/Injected/MAB_siblings/**/Original - Copy'    
    
elif condition == 'Injected' and direction == 'Left' and pheno == 'MAB_phenos':    
    Experiment_condition = '/Injected/MAB_phenos/**/Original - Copy' 
    
elif condition == 'Injected' and direction == 'Right' and pheno == 'MAB_phenos':    
    Experiment_condition = '/Injected/MAB_phenos/**/Original - Copy'    
    
else: 
    print('Error: Condition and direction not defined')      


#%% Paths
# for all jpng image files called left cropped
img_paths_left = glob.glob(Experiment_folder + 
                              Experiment_condition + '/Left_8x' + '**' + '.tif', recursive=False )

# # for all jpng img files called right cropped
img_paths_right = glob.glob(Experiment_folder + 
                              Experiment_condition + '/Right_8x' + '**' +'.tif', recursive=False)

# Set empty list for later
left_eye_features = np.zeros((len(img_paths_left), 9))

right_eye_features = np.zeros((len(img_paths_right), 9))

eye_parameters_all = pd.DataFrame([])

#%% Load images

if direction == 'Left': 
    filename = 'image_paths_left'
    img_paths= img_paths_left
    
elif direction == 'Right':   
    filename = 'image_paths_right'
    img_paths= img_paths_right
    
else: 
    print ('error, direction not defined')   
    
#--------------------- 
# for left side images       
for f, fish_imgs in enumerate(img_paths):
    img= Image.open(fish_imgs) 
    #img = io.imread(fish_imgs)
    fig, ax = plt.subplots()
    plt.imshow(img)
    plt.show()    
        

#%% # manipulate images 

    # crop image to get rid of uneccessary background 
    # img_manip.crop_around_center(img, length, height)    
    # orgininal image size =>  height = 1944 and length = 2592
    # cropped_img = img_manip.crop_around_center(img, 6600, 4700)
    
    # # plot image
    # fig, ax = plt.subplots()
    # plt.imshow(cropped_img)
    # plt.title('cropped_image')
    # plt.show()    
    
    # gray_img = color.rgb2gray(cropped_img[:,:,:3])

    # brighten image
    #PIL_image = Image.fromarray(img, 'RGB')
    
    fish_ID= img_paths_left[f].split('\\')[1]
    print('Fish_ID= ' + fish_ID)
    
    brighten = ImageEnhance.Brightness(img)
    bright_img = brighten.enhance(3.8)
    # bright_img.show()
    
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(bright_img)
    plt.title('brightened_image')
    plt.show()  
    
    
    # increase the contrast of the image 
    up_contrast = ImageEnhance.Contrast(bright_img)
    contrast_img = up_contrast.enhance(4.15)
    # contrast_img.show()
    
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(contrast_img)
    plt.title('upped_contrast_image')
    plt.show()  
    
    
    brighten_2 = ImageEnhance.Brightness(contrast_img)
    bright_img_2 = brighten_2.enhance(200)
    # bright_img_2.show()
    
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(bright_img_2)
    plt.title('brightened_2_image')
    plt.show()  
     
        
    up_contrast_2 = ImageEnhance.Contrast(bright_img_2)
    contrast_img_2 = up_contrast_2.enhance(1.1)
    # contrast_img_2.show()
    
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(contrast_img_2)
    plt.title('contrasted_2_image')
    plt.show()  
    
    
    contrast_img_2.save(fish_imgs[:-4]  + '_enhanced.png' )
    img2 = io.imread(fish_imgs[:-4] + '_enhanced.png')
    
    
    # convert image to grayscale
    gray_img = color.rgb2gray(img2[:,:,:3])
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(gray_img, cmap='gray')
    plt.title('gray_img')
    plt.show()
    
    # # denoise the image via mdeian filter 
    # denoised_img = filters.median(gray_img)
    
    # # plot image
    # fig, ax = plt.subplots()
    # plt.imshow(denoised_img, cmap='gray')
    # plt.title('denoised_image')
    # plt.show()
    
    # # sharpen image with unsharp mask filter with strel (strucutral element )
    # #  The sharp details are identified as a difference between the original image and its blurred version. 
    # sharp_img = unsharp_mask(gray_img, radius=2, amount=10)
    
    # # plot image
    # fig, ax = plt.subplots()
    # plt.imshow(sharp_img, cmap='gray')
    # plt.title('sharpened_image')
    # plt.show()
  
    
#%%

    # create a histogram of the blurred grayscale image
    histogram, bin_edges = np.histogram(gray_img, bins=256, range=(0.0, 1.0))
    
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
    eye_threshold = threshold_multiotsu(gray_img, classes=5)
    eye_threshold = (eye_threshold[0]*.25)
    # eye_threshold = skimage.filters.threshold_otsu(sharp_img)
    # eye_threshold = threshold_minimum(sharp_img)
    # eye_threshold = threshold_yen(sharp_img)
        
    ## If the regions of interests is darker than the background use $ gray_image > t
    ## If the regions interests is lighter than the background use $ gray_image < t
    eye_mask = gray_img < eye_threshold

    # plot image
    fig, ax = plt.subplots()
    plt.imshow(eye_mask, cmap='gray')
    plt.title('eye_mask')
    plt.show()
    
#-------- 
 
    # remove small particles from image
    cleaned_img_eye_1 = remove_small_objects(eye_mask, min_size=110, connectivity=4)
    
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(cleaned_img_eye_1, cmap='gray')
    plt.title('cleaned_eye_mask')
    plt.show()

#--------    
  
    # # dilate image to close gaps
    # footprint = disk(1.0)
    # dilated_eye_mask = binary_dilation(cleaned_img_eye_1, footprint)
        
    # # plot image
    # fig, ax = plt.subplots()
    # plt.imshow(dilated_eye_mask, cmap='gray')
    # plt.title('dilated_eye_mask')
    # plt.show()

#-------- 
 
    # remove small particles from image
    cleaned_img_eye_2 = remove_small_objects(cleaned_img_eye_1, min_size=4400, connectivity=2)
    
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(cleaned_img_eye_2, cmap='gray')
    plt.title('cleaned_dilated_mask')
    plt.show()
   
#--------  
 # Label areas in mask
    labels = measure.label(cleaned_img_eye_2)
    
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
        
    
    # =-----------------
       
    ## centroid: array --> Centroid coordinate tuple (row, col).(row, col)
    ## area: int --> Number of pixels in region
    ## perimeter: float --> Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
    ## orientation: float --> Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.
    ## axis_minor_length: float --> The length of the minor axis of the ellipse that has the same normalized second central moments as the region   
    ## axis_major_length: float --> The length of the major axis of the ellipse that has the same normalized second central moments as the region.    
    
    props_2 = measure.regionprops_table(labels, properties=( 'centroid',
                                                              'area',
                                                              'perimeter',
                                                              'orientation',
                                                              'major_axis_length',
                                                              'minor_axis_length',
                                                              'label',
                                                              'eccentricity'            ) )
    
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
        
    
  
    eye_parameters_all= pd.concat([eye_parameters_all, eye_properties], axis=1)



#%%   # Summarise findings on one table 


# # Control
if condition== 'Control' and direction== 'Left' and pheno == 'MAB_phenos':   
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
            'Direction fish facing: ' + direction)  
    left_eye_parameters_all = pd.DataFrame(eye_parameters_all).T
    left_eye_parameters_all= left_eye_parameters_all.reset_index(drop=True)
    
    
    left_eye_parameters_all.to_csv( str(Path(img_paths_left[f]).parents[4]) 
                                  + '/' + condition + '/' + pheno 
                                  + '/MAB_phenos_Left_Eye_Parameters.csv' , index=False)
        
    
elif condition== 'Control' and direction== 'Right' and pheno == 'MAB_phenos':  
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
            'Direction fish facing: ' + direction)  
    right_eye_parameters_all = pd.DataFrame(eye_parameters_all).T
    right_eye_parameters_all= right_eye_parameters_all.reset_index(drop=True)
    
    
    right_eye_parameters_all.to_csv( str(Path(img_paths_right[f]).parents[4]) 
                                  + '/' + condition + '/' + pheno  
                                  + '/MAB_phenos_Right_Eye_Parameters.csv' , index=False)

    
    
elif condition== 'Control' and direction== 'Left' and pheno == 'MAB_siblings':  
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
            'Direction fish facing: ' + direction)  
    left_eye_parameters_all = pd.DataFrame(eye_parameters_all).T
    left_eye_parameters_all= left_eye_parameters_all.reset_index(drop=True)
    
    
    left_eye_parameters_all.to_csv( str(Path(img_paths_left[f]).parents[4]) 
                                  + '/' + condition + '/' + pheno 
                                  + '/MAB_siblings_Left_Eye_Parameters.csv' , index=False)

    
elif condition=='Control' and direction== 'Right' and pheno == 'MAB_siblings':  
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
            'Direction fish facing: ' + direction)  
    right_eye_parameters_all = pd.DataFrame(eye_parameters_all).T
    right_eye_parameters_all= right_eye_parameters_all.reset_index(drop=True)
    
    
    right_eye_parameters_all.to_csv( str(Path(img_paths_right[f]).parents[4]) 
                                  + '/' + condition + '/' + pheno 
                                  + '/MAB_siblings_Right_Eye_Parameters.csv' , index=False) 
    
#------------------    
# # Injected
elif condition== 'Injected' and direction== 'Left' and pheno == 'MAB_phenos':   
    print('one')
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
            'Direction fish facing: ' + direction)  
    left_eye_parameters_all = pd.DataFrame(eye_parameters_all).T
    left_eye_parameters_all= left_eye_parameters_all.reset_index(drop=True)
    
    
    left_eye_parameters_all.to_csv( str(Path(img_paths_left[f]).parents[4]) 
                                  + '/' + condition + '/' + pheno 
                                  + '/MAB_phenos_Left_Eye_Parameters.csv' , index=False)
        
    
elif condition== 'Injected' and direction== 'Right' and pheno == 'MAB_phenos':  
    print('two')
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
            'Direction fish facing: ' + direction)  
    right_eye_parameters_all = pd.DataFrame(eye_parameters_all).T
    right_eye_parameters_all= right_eye_parameters_all.reset_index(drop=True)
    
    
    right_eye_parameters_all.to_csv( str(Path(img_paths_right[f]).parents[4]) 
                                  + '/' + condition + '/' + pheno 
                                  + '/MAB_phenos_Right_Eye_Parameters.csv' , index=False)

     
elif condition== 'Injected' and direction== 'Left' and pheno == 'MAB_siblings': 
    print('three')
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
            'Direction fish facing: ' + direction)  
    left_eye_parameters_all = pd.DataFrame(eye_parameters_all).T
    left_eye_parameters_all= left_eye_parameters_all.reset_index(drop=True)
    
    
    left_eye_parameters_all.to_csv( str(Path(img_paths_left[f]).parents[4]) 
                                  + '/' + condition + '/' + pheno 
                                  + '/MAB_siblings_Left_Eye_Parameters.csv' , index=False)

    
elif condition=='Injected' and direction== 'Right' and pheno == 'MAB_siblings': 
    print('four')
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
            'Direction fish facing: ' + direction)  
    right_eye_parameters_all = pd.DataFrame(eye_parameters_all).T
    right_eye_parameters_all= right_eye_parameters_all.reset_index(drop=True)
    
    
    right_eye_parameters_all.to_csv( str(Path(img_paths_right[f]).parents[4]) 
                                  + '/' + condition + '/' + pheno 
                                  + '/MAB_siblings_Right_Eye_Parameters.csv' , index=False)
#------------------     
    
else:
    print ('Error saving csv data?')  
          