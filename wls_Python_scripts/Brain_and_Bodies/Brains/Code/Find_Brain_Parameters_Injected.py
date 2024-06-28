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

    
#%% Search through folder/s to find image files 

Experiment_folder = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_TL_Brains_and_Bodies/Testing_area/Brain_images/'

# # Set flags
# condition = 'Control'
condition = 'Injected' 

direction = 'Left'
# direction = 'Right'
    
#%% Search through folder/s to find image files 

# # Control --------
# # ------------------    
if condition == 'Control' and direction == 'Left':
    
    # Set experiment condition to be analysed '
    Experiment_condition = condition
       
    # Find images in experiment folder
    img_paths = glob.glob(Experiment_folder + Experiment_condition + '/**' + '/Original - Copy/'
                                + 'Substack' + ' **' + '.tif', recursive=False )

    print ('Condition: ' + condition + ', ' + 'Direction fish facing: ' + direction)  

#-----------
elif condition == 'Control' and direction == 'Right':
    
    # Set experiment condtion to be analysed '
    Experiment_condition = condition
       
    # Find images in experiment folder
    img_paths = glob.glob(Experiment_folder + Experiment_condition + '/**' + '/Original - Copy/'
                                + 'Substack' + ' **' + '.tif', recursive=False )
    
    print ('Condition: ' + condition + ', ' + 'Direction fish facing: ' + direction)  
    
# # Injected -------
# # ------------------    
elif condition == 'Injected' and direction == 'Left':
    
    # Set experiment condtion to be analysed '
    Experiment_condition = condition
       
    # Find images in experiment folder
    img_paths = glob.glob(Experiment_folder + Experiment_condition + '/**' + '/Original - Copy/'
                                + 'Substack' + ' **' + '.tif', recursive=False )


    print ('Condition: ' + condition + ', ' + 'Direction fish facing: ' + direction)  
   
#-----------    
elif condition == 'Injected' and direction == 'Right':
    
    # Set experiment condtion to be analysed '
    Experiment_condition = condition
       
    # Find images in experiment folder
    img_paths = glob.glob(Experiment_folder + Experiment_condition + '/**' + '/Original - Copy/'
                                + 'Substack' + ' **' + '.tif', recursive=False )

    
    print ('Condition: ' + condition + ', ' + 'Direction fish facing: ' + direction)  

#-----------             
else: 
    print('Error: Condition and direction not defined')    
    
#%% Define empty dataframes to store fish lengths

Fish_Whole_Brain_Lengths_all     = pd.DataFrame([])
Fish_Forebrain_Lengths_all       = pd.DataFrame([])
Fish_Midbrain_Lengths_all        = pd.DataFrame([])
Fish_Hindbrain_Lengths_all       = pd.DataFrame([])

Fish_Forebrain_Widths_all       = pd.DataFrame([])
Fish_Midbrain_Widths_all        = pd.DataFrame([])
Fish_Hindbrain_Widths_all       = pd.DataFrame([])

Fish_IDs_all                    = pd.DataFrame([])

#%%  Analyse fish lengths

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
    cleaned_mask = remove_small_objects(mask, min_size=1, connectivity=1)
    
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
        
    elif direction == 'Right':
        centroids_table2= centroids_table.sort_values('x_cordinates', ascending=False)
        centroids_table2= centroids_table2.reset_index(drop=True)
        
    else: 
        print('direction not determined')
    
    
    xy_positions= pd.DataFrame([])

  
    for x ,y in zip(centroids_table2['x_cordinates'], centroids_table2['y_cordinates']):
        z= pd.DataFrame([[(x,y)]])
        # print(z)
        xy_positions= pd.concat([xy_positions, z], axis=0)
    
    xy_positions= xy_positions.reset_index(drop=True)
    xy_positions.columns=['xy postions']

    # Define x,y positions of brain areas/landmarks
    forebrain_start = xy_positions['xy postions'][0]
    
    if xy_positions['xy postions'][1][1] < xy_positions['xy postions'][2][1]:
        forebrain_upper = xy_positions['xy postions'][1]
        forbrain_lower = xy_positions['xy postions'][2]
    else:
        forebrain_upper = xy_positions['xy postions'][2]
        forbrain_lower = xy_positions['xy postions'][1]
            
    forebrain_end_midbrain_start = ( ((xy_positions['xy postions'][3][0] + xy_positions['xy postions'][4][0]) / 2),
                                    ((xy_positions['xy postions'][3][1] + xy_positions['xy postions'][4][1]) / 2) )
    
    if xy_positions['xy postions'][5][1] < xy_positions['xy postions'][6][1]:
        midbrain_upper = xy_positions['xy postions'][5]
        midbrain_lower = xy_positions['xy postions'][6]
    else:
        midbrain_upper = xy_positions['xy postions'][6]
        midbrain_lower = xy_positions['xy postions'][5]   

    midbrain_end_hindbrain_start = ( ((xy_positions['xy postions'][7][0] + xy_positions['xy postions'][8][0]) / 2),
                                    ((xy_positions['xy postions'][7][1] + xy_positions['xy postions'][8][1]) / 2) )
     
    if xy_positions['xy postions'][9][1] < xy_positions['xy postions'][10][1]:
        hindbrain_upper = xy_positions['xy postions'][9]
        hindbrain_lower = xy_positions['xy postions'][10]
    else:
        hindbrain_upper = xy_positions['xy postions'][10]
        hindbrain_lower = xy_positions['xy postions'][9]  

    hind_brain_end = xy_positions['xy postions'][11]
        
        
    # Find Brain Lengths
    forebrain_length= ( math.sqrt(((forebrain_start[0] - forebrain_end_midbrain_start[0])**2) 
                     + (forebrain_start[1] - forebrain_end_midbrain_start[1])**2) )
    forebrain_length = pd.DataFrame([forebrain_length])
    
    midbrain_length= ( math.sqrt(((forebrain_end_midbrain_start[0] - midbrain_end_hindbrain_start[0])**2) 
                     + (forebrain_end_midbrain_start[1] - midbrain_end_hindbrain_start[1])**2) )
    midbrain_length = pd.DataFrame([midbrain_length])
    
    hindbrain_length= ( math.sqrt(((midbrain_end_hindbrain_start[0] - hind_brain_end[0])**2) 
                     + (midbrain_end_hindbrain_start[1] - hind_brain_end[1])**2) )
    hindbrain_length = pd.DataFrame([hindbrain_length])
            
    whole_brain_length= ( math.sqrt(((forebrain_start[0] - hind_brain_end[0])**2) 
                     + (forebrain_start[1] - hind_brain_end[1])**2) )
    whole_brain_length = pd.DataFrame([whole_brain_length])
    
      
    # Find Brain Widths  
    forebrain_width = ( math.sqrt(((forebrain_upper[0] - forbrain_lower[0])**2) 
                     + (forebrain_upper[1] - forbrain_lower[1])**2) )
    forebrain_width = pd.DataFrame([forebrain_width])
    
    midbrain_width = ( math.sqrt(((midbrain_upper[0] - midbrain_lower[0])**2) 
                     + (midbrain_upper[1] - midbrain_lower[1])**2) )
    midbrain_width = pd.DataFrame([midbrain_width])
    
    hindbrain_width = ( math.sqrt(((hindbrain_upper[0] - hindbrain_lower[0])**2) 
                     + (hindbrain_upper[1] - hindbrain_lower[1])**2) )
    hindbrain_width = pd.DataFrame([hindbrain_width])
    
    
    fish_ID= pd.DataFrame([fish_ID])
      
    # Concatinate Brain Measuremetns from each fish 
    Fish_Forebrain_Lengths_all       = pd.concat([Fish_Forebrain_Lengths_all, forebrain_length])
    Fish_Midbrain_Lengths_all        = pd.concat([Fish_Midbrain_Lengths_all, midbrain_length])
    Fish_Hindbrain_Lengths_all       = pd.concat([Fish_Hindbrain_Lengths_all, hindbrain_length])
    Fish_Whole_Brain_Lengths_all     = pd.concat([Fish_Whole_Brain_Lengths_all, whole_brain_length])

    Fish_Forebrain_Widths_all        = pd.concat([Fish_Forebrain_Widths_all, forebrain_width])
    Fish_Midbrain_Widths_all         = pd.concat([Fish_Midbrain_Widths_all, midbrain_width])
    Fish_Hindbrain_Widths_all        = pd.concat([Fish_Hindbrain_Widths_all, hindbrain_width])

    Fish_IDs_all                    = pd.concat([Fish_IDs_all, fish_ID ])
    
Data= pd.concat([Fish_IDs_all,
                 Fish_Forebrain_Lengths_all, 
                 Fish_Midbrain_Lengths_all, 
                 Fish_Hindbrain_Lengths_all, 
                 Fish_Whole_Brain_Lengths_all,
                 Fish_Forebrain_Widths_all, 
                 Fish_Midbrain_Widths_all, 
                 Fish_Hindbrain_Widths_all ], axis=1)

Data.columns=['Fish_IDs',
                 'Forebrain_Lengths', 
                 'Midbrain_Lengths', 
                 'Hindbrain_Lengths', 
                 'Whole_Brain_Lengths',
                 'Forebrain_Widths', 
                 'Midbrain_Widths', 
                 'Hindbrain_Widths' ]  


#%%

if condition=='Control' and direction== 'Left':   
    Data.to_csv( str(Path(img_paths[f]).parents[2]) + '/' + condition + '_Brain_Parameters.csv', index=False)


elif condition=='Control' and direction== 'Right':   
    Data.to_csv( str(Path(img_paths[f]).parents[2]) + '/' + condition + '_Brain_Parameters.csv', index=False)
       
    
elif condition=='Injected' and direction== 'Left':   
    Data.to_csv( str(Path(img_paths[f]).parents[2]) + '/' + condition + '_Brain_Parameters.csv', index=False)
       
    
elif condition=='Injected' and direction== 'Right':   
    Data.to_csv( str(Path(img_paths[f]).parents[2]) + '/' + condition + '_Brain_Parameters.csv', index=False)
       
    
else:
    print ('Error saving csv data')  
    
   
#%%  Finish 


