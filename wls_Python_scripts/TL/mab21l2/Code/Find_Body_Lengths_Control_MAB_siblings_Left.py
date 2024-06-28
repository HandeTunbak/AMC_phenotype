# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Dr Hande Tunbak)
"""

#%% Import Libraries
#-------------------------------
#-------------------------------

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

    
#%% Search through folder/s to find image files 
#-------------------------------
#-------------------------------

Experiment_folder = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/'

# # Set flags
condition = 'Control'
# condition = 'Injected' 

direction = 'Left'
# direction = 'Right'

pheno = 'MAB_siblings'
# pheno = 'MAB_phenos'

    
#%% Search through folder/s to find image files 

# # Control --------
# # ------------------    
if condition == 'Control' and direction == 'Left' and pheno == 'MAB_siblings':
    
    # Set experiment condition to be analysed '
    Experiment_condition = condition
       
    # Find images in experiment folder
    img_paths = glob.glob(Experiment_folder + Experiment_condition + '/MAB_siblings' +
                          '/**' + '/Original - Copy' + 
                          '/Left_3x' + '**' + '.tif', recursive=False )

    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  

#-----------
elif condition == 'Control' and direction == 'Right' and pheno == 'MAB_siblings':
    
    # Set experiment condition to be analysed '
    Experiment_condition = condition
       
    # Find images in experiment folder
    img_paths = glob.glob(Experiment_folder + Experiment_condition + '/MAB_siblings' +
                          '/**' + '/Original - Copy' + 
                          '/Right_3x' + '**' + '.tif', recursive=False )

    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  

#-----------
elif condition == 'Control' and direction == 'Left' and pheno == 'MAB_phenos':
    
    # Set experiment condition to be analysed '
    Experiment_condition = condition
       
    # Find images in experiment folder
    img_paths = glob.glob(Experiment_folder + Experiment_condition + '/MAB_phenos' +
                          '/**' + '/Original - Copy' + 
                          '/Left_3x' + '**' + '.tif', recursive=False )

    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  

#-----------
elif condition == 'Control' and direction == 'Right' and pheno == 'MAB_phenos':
    
    # Set experiment condition to be analysed '
    Experiment_condition = condition
       
    # Find images in experiment folder
    img_paths = glob.glob(Experiment_folder + Experiment_condition + '/MAB_phenos' +
                          '/**' + '/Original - Copy' + 
                          '/Right_3x' + '**' + '.tif', recursive=False )

    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  


#---------------  
# # Injected -------
# # ------------------    
elif condition == 'Injected' and direction == 'Left' and pheno == 'MAB_siblings':
    
    # Set experiment condition to be analysed '
    Experiment_condition = condition
       
    # Find images in experiment folder
    img_paths = glob.glob(Experiment_folder + Experiment_condition + '/MAB_siblings' +
                          '/**' + '/Original - Copy' + 
                          '/Left_3x' + '**' + '.tif', recursive=False )

    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  


#-----------    
elif condition == 'Injected' and direction == 'Right' and pheno == 'MAB_siblings':
    
    # Set experiment condition to be analysed '
    Experiment_condition = condition
       
    # Find images in experiment folder
    img_paths = glob.glob(Experiment_folder + Experiment_condition + '/MAB_siblings' +
                          '/**' + '/Original - Copy' + 
                          '/Right_3x' + '**' + '.tif', recursive=False )

    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  

#-----------
elif condition == 'Injected' and direction == 'Left' and pheno == 'MAB_phenos':
    
    # Set experiment condition to be analysed '
    Experiment_condition = condition
       
    # Find images in experiment folder
    img_paths = glob.glob(Experiment_folder + Experiment_condition + '/MAB_phenos' +
                          '/**' + '/Original - Copy' + 
                          '/Left_3x' + '**' + '.tif', recursive=False )

    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  
   
#-----------    
elif condition == 'Injected' and direction == 'Right' and pheno == 'MAB_phenos':
    
    # Set experiment condition to be analysed '
    Experiment_condition = condition
       
    # Find images in experiment folder
    img_paths = glob.glob(Experiment_folder + Experiment_condition + '/MAB_phenos' +
                          '/**' + '/Original - Copy' + 
                          '/Right_3x' + '**' + '.tif', recursive=False )

    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  

#-----------             
else: 
    print('Error: Condition and direction not defined')    
    
    
#%% Define empty dataframes to store fish lengths

Fish_head_lengths_all           = pd.DataFrame([])
Fish_tail_lengths_all           = pd.DataFrame([])
Fish_body_lengths_all           = pd.DataFrame([])
Fish_head_body_ratios_all       = pd.DataFrame([])
Fish_IDs_all                    = pd.DataFrame([])




for f, fish_imgs in enumerate(img_paths):
    
    # Set Fish_ids
    fish_ID= img_paths[f].split('\\')[1]
    print('\n' + 'Fish_ID= ' + fish_ID)
        
    # Load image
    img = cv2.imread(fish_imgs)
    plt.imshow(img)
    
    ## Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    ## Mask of green (36,25,25) ~ (86, 255,255) --> could have been mask (or threshold) = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    threshold_range = cv2.inRange(hsv, (40, 25, 25), (80, 255,255)) # lower and upper values
    # plt.imshow(threshold_range)
    
    ## Convert image to greyscale for thresholding 
    gray_img = color.rgb2gray(img[:,:,:3])
    
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(gray_img, cmap='gray')
    plt.title('gray_img')
    plt.show()
    
    ## Create mask with format of array of bools
    mask = gray_img < threshold_range
    # plt.imshow(mask)
    
    # remove small particles from masked image incase the mask is not great
    cleaned_mask = remove_small_objects(mask, min_size=200, connectivity=4)
    
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(cleaned_mask, cmap='gray')
    plt.title('cleaned_eye_mask')
    plt.show()
    
    ## Create binary image of cleaned mask
    labels = measure.label(cleaned_mask)
    
    # Set image to trace over - tracing over the binary image
    fig = px.imshow(cleaned_mask, binary_string=True)
    fig.update_traces(hoverinfo='skip') 
    
    # hover is only for label info
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
    
    if direction == 'Left':
        centroids_table2= centroids_table.sort_values('x_cordinates', ascending=True)
        centroids_table2= centroids_table2.reset_index(drop=True)
        centroids_table2= centroids_table2.drop([1])
        
    elif direction == 'Right':
        centroids_table2= centroids_table.sort_values('x_cordinates', ascending=False)
        centroids_table2= centroids_table2.reset_index(drop=True)
        centroids_table2= centroids_table2.drop([1])
        
    else: 
        print('direction not determined')
    
    
    xy_positions= pd.DataFrame([])
    
    for x ,y in zip(centroids_table2['x_cordinates'], centroids_table2['y_cordinates']):
        z= pd.DataFrame([[(x,y)]])
        # print(z)
        xy_positions= pd.concat([xy_positions, z], axis=0)
    
    xy_positions= xy_positions.reset_index(drop=True)
    xy_positions.columns=['xy postions']
    
    # # Find fish length    
    distances_all= pd.DataFrame([])
    
    for i in range(len(xy_positions)-1): 
        current_x_value= xy_positions['xy postions'][i][0]
        next_x_value= xy_positions['xy postions'][i+1][0]
        # print(current_x_value)
        # print(next_x_value)
        
        current_y_value= xy_positions['xy postions'][i][1]
        next_y_value= xy_positions['xy postions'][i+1][1]
        # print(current_y_value)
        # print(next_y_value)
        
        distances = [math.sqrt(((next_x_value-current_x_value)**2)+ 
                               ((next_y_value-current_y_value)**2))]
        #print(distances)
        
        distances= pd.DataFrame([distances])
        distances_all= pd.concat([distances_all, distances])
    
    if len(distances_all) ==7:
        print ('measurements: 7 of 7' )
    else: 
        print ('measurements: '+ str(len(distances_all))  + ' of 7' )
    
    fish_head_length= sum(distances_all[0][0:2])
    fish_head_length= pd.DataFrame([fish_head_length])
    
    fish_tail_length= sum(distances_all[0][2:7])
    fish_tail_length= pd.DataFrame([fish_tail_length])
    
    fish_body_length= sum(distances_all[0])
    fish_body_length= pd.DataFrame([fish_body_length])
    
    fish_head_body_ratio= fish_head_length/fish_tail_length
    
    Fish_head_lengths_all= pd.concat([Fish_head_lengths_all, fish_head_length])
    Fish_tail_lengths_all= pd.concat([Fish_tail_lengths_all, fish_tail_length])
    Fish_body_lengths_all= pd.concat([Fish_body_lengths_all, fish_body_length])
    Fish_head_body_ratios_all= pd.concat([Fish_head_body_ratios_all, fish_head_body_ratio])
    
    fish_ID= pd.DataFrame([fish_ID])
    Fish_IDs_all= pd.concat([Fish_IDs_all, fish_ID], axis=0)
    

Fish_head_lengths_all.columns=['Fish Head Length']
Fish_head_lengths_all= Fish_head_lengths_all.reset_index(drop= True)

Fish_tail_lengths_all.columns=['Fish Tail Length']
Fish_tail_lengths_all= Fish_tail_lengths_all.reset_index(drop= True)

Fish_body_lengths_all.columns=['Fish Body Length']
Fish_body_lengths_all= Fish_body_lengths_all.reset_index(drop= True)

Fish_head_body_ratios_all.columns=['Fish Head/Body ratio']
Fish_head_body_ratios_all= Fish_head_body_ratios_all.reset_index(drop= True)

Fish_IDs_all.columns=['Fish_IDs']
Fish_IDs_all= Fish_IDs_all.reset_index(drop= True)

       
Fish_lengths_all = pd.concat( [Fish_IDs_all, Fish_head_lengths_all, 
                               Fish_tail_lengths_all, Fish_body_lengths_all, 
                               Fish_head_body_ratios_all ], axis=1 )
    
#%% Save Body Lengths
#-------------------------------
#-------------------------------

# # Control
if condition== 'Control' and direction== 'Left' and pheno == 'MAB_phenos':   
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  
    Fish_lengths_all.to_csv( str(Path(img_paths[f]).parents[4]) 
                                 + '/' + condition + '/' + pheno 
                                 + '/MAB_phenos_Left_Body_Length.csv' , index=False)
        
    
elif condition== 'Control' and direction== 'Right' and pheno == 'MAB_phenos':   
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  
    Fish_lengths_all.to_csv( str(Path(img_paths[f]).parents[4]) 
                                 + '/' + condition + '/' + pheno 
                                 + '/MAB_phenos_Right_Body_Length.csv' , index=False)

     
elif condition== 'Control' and direction== 'Left' and pheno == 'MAB_siblings':  
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  
    Fish_lengths_all.to_csv( str(Path(img_paths[f]).parents[4]) 
                                 + '/' + condition + '/' + pheno 
                                 + '/MAB_siblings_Left_Body_Length.csv' , index=False)

    
elif condition=='Control' and direction== 'Right' and pheno == 'MAB_siblings':   
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  
    Fish_lengths_all.to_csv( str(Path(img_paths[f]).parents[4]) 
                                 + '/' + condition + '/' + pheno 
                                 + '/MAB_siblings_Right_Body_Length.csv' , index=False)

#------------------    
# # Injected
elif condition=='Injected' and direction== 'Left' and pheno == 'MAB_phenos':   
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  
    Fish_lengths_all.to_csv( str(Path(img_paths[f]).parents[4]) 
                                 + '/' + condition + '/' + pheno 
                                 + '/MAB_phenos_Left_Body_Length.csv' , index=False)

elif condition=='Injected' and direction== 'Right' and pheno == 'MAB_phenos':  
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  
    Fish_lengths_all.to_csv( str(Path(img_paths[f]).parents[4]) 
                                 + '/' + condition + '/' + pheno 
                                 + '/MAB_phenos_Right_Body_Length.csv' , index=False)
    
elif condition=='Injected' and direction== 'Left' and pheno == 'MAB_siblings': 
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  
    Fish_lengths_all.to_csv( str(Path(img_paths[f]).parents[4]) 
                                 + '/' + condition + '/' + pheno 
                                 + '/MAB_siblings_Left_Body_Length.csv' , index=False)
 
elif condition=='Injected' and direction== 'Right' and pheno == 'MAB_siblings':   
    print ('Condition: ' + condition + ', ' + 'Pheno: ' + pheno + ', '
           'Direction fish facing: ' + direction)  
    Fish_lengths_all.to_csv( str(Path(img_paths[f]).parents[4]) 
                                 + '/' + condition + '/' + pheno 
                                 + '/MAB_siblings_Right_Body_Length.csv' , index=False)
    
#------------------     
    
else:
    print ('Error saving csv data?')  
        
#%% # Finish


#%%        
        
        
        