# -*- coding: utf-8 -*-
#%%
"""
Created on Tue Mar  1 18:01:10 2022

@author: hande
"""

''' This code is to analyse the legnths and width of fish not treated in PTU. '''


#%% # Read if first time running this code!
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

''' for first time use you will need to install
> seaborn:        $ conda install seaborn                                          - click y when prompted
 '''

#%% # Import libraries
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

import os
import sys
import glob
import cv2
import numpy as np
import scipy as sp
import imageio
import matplotlib.pyplot as plt
import pandas as pd
import math
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path
import seaborn as sns
from scipy import stats

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "svg"

from skimage import io, color, data, filters, measure, morphology, img_as_float, exposure, restoration
from skimage.color import rgb2gray
from skimage.filters import unsharp_mask, threshold_multiotsu, threshold_isodata, threshold_otsu, threshold_minimum, threshold_yen
from skimage.filters.rank import autolevel, enhance_contrast
from skimage.morphology import disk, diamond,  white_tophat, binary_dilation, remove_small_objects, label


#%% # Set paths for csv files  
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Location with subdirectories
Experiment_folder = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_TL_Brains_and_Bodies/Testing_area/Body_images_for_brains/New- repeated'

#------------------------------
# # Control
Control_Left_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/Control_Left_Eye_Parameters.csv') 

Control_Right_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/Control_Right_Eye_Parameters.csv')  

Control_Left_Body_Parameters_csv= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/Control_Left_Body_Parameters.csv') 

Control_Right_Body_Parameters_csv= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/Control_Right_Body_Parameters.csv') 

# #     Control Eccentricity
Control_Left_eye_eccentricity= Control_Left_eye_paramaters['eccentricity']
Control_Right_eye_eccentricity= Control_Right_eye_paramaters['eccentricity']
Combined_Control_eye_eccentricity = pd.concat([Control_Left_eye_eccentricity, 
                                               Control_Right_eye_eccentricity])

# #     Control Eye Area
Control_Left_eye_area= Control_Left_eye_paramaters['area']
Control_Right_eye_area= Control_Right_eye_paramaters['area']
Combined_Control_eye_area = pd.concat([Control_Left_eye_area, 
                                               Control_Right_eye_area])

# #     Control Body Length
Control_Left_Body_Length= Control_Left_Body_Parameters_csv['Fish Body Length']
Control_Right_Body_Length= Control_Right_Body_Parameters_csv['Fish Body Length']
Combined_Control_Body_Length= pd.concat([Control_Left_Body_Length, 
                                               Control_Right_Body_Length])

# #     Control Head Length
Control_Left_Head_Length= Control_Left_Body_Parameters_csv['Fish Head Length']
Control_Right_Head_Length= Control_Right_Body_Parameters_csv['Fish Head Length']
Combined_Control_Head_Length= pd.concat([Control_Left_Head_Length, 
                                               Control_Right_Head_Length])

# #     Control Tail Length
Control_Left_Tail_Length= Control_Left_Body_Parameters_csv['Fish Tail Length']
Control_Right_Tail_Length= Control_Right_Body_Parameters_csv['Fish Tail Length']
Combined_Control_Tail_Length= pd.concat([Control_Left_Tail_Length, 
                                               Control_Right_Tail_Length])

# #     Control Eye Major Axis Length
Control_Left_eye_major_axis_length= Control_Left_eye_paramaters['major_axis_length']
Control_Right_eye_major_axis_length= Control_Right_eye_paramaters['major_axis_length']
Combined_Control_eye_major_axis_length = pd.concat([Control_Left_eye_major_axis_length, 
                                               Control_Right_eye_major_axis_length])


#------------------------------
# # Injected
Injected_Left_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/Injected_Left_Eye_Parameters.csv') 

Injected_Right_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/Injected_Right_Eye_Parameters.csv')  

Injected_Left_Body_Parameters_csv= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/Injected_Left_Body_Parameters.csv') 

Injected_Right_Body_Parameters_csv= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/Injected_Right_Body_Parameters.csv') 


# #     Injected Eccentricity
Injected_Left_eye_eccentricity= Injected_Left_eye_paramaters['eccentricity']
Injected_Right_eye_eccentricity= Injected_Right_eye_paramaters['eccentricity']
Combined_Injected_eye_eccentricity = pd.concat([Injected_Left_eye_eccentricity, 
                                               Injected_Right_eye_eccentricity])

# #     Injected Eye Area
Injected_Left_eye_area= Injected_Left_eye_paramaters['area']
Injected_Right_eye_area= Injected_Right_eye_paramaters['area']
Combined_Injected_eye_area = pd.concat([Injected_Left_eye_area, 
                                               Injected_Right_eye_area])

# #     Injected Body Length
Injected_Left_Body_Length= Injected_Left_Body_Parameters_csv['Fish Body Length']
Injected_Right_Body_Length= Injected_Right_Body_Parameters_csv['Fish Body Length']
Combined_Injected_Body_Length= pd.concat([Injected_Left_Body_Length, 
                                               Injected_Right_Body_Length])

# #     Injected Head Length
Injected_Left_Head_Length= Injected_Left_Body_Parameters_csv['Fish Head Length']
Injected_Right_Head_Length= Injected_Right_Body_Parameters_csv['Fish Head Length']
Combined_Injected_Head_Length= pd.concat([Injected_Left_Head_Length, 
                                               Injected_Right_Head_Length])

# #     Injected Tail Length
Injected_Left_Tail_Length= Injected_Left_Body_Parameters_csv['Fish Tail Length']
Injected_Right_Tail_Length= Injected_Right_Body_Parameters_csv['Fish Tail Length']
Combined_Injected_Tail_Length= pd.concat([Injected_Left_Tail_Length, 
                                               Injected_Right_Tail_Length])

# #     Injected Eye Major Axis Length
Injected_Left_eye_major_axis_length= Injected_Left_eye_paramaters['major_axis_length']
Injected_Right_eye_major_axis_length= Injected_Right_eye_paramaters['major_axis_length']
Combined_Injected_eye_major_axis_length = pd.concat([Injected_Left_eye_major_axis_length, 
                                               Injected_Right_eye_major_axis_length])


#%% # Prep Data (various)
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# # Conversions from pixels to 1mm length
Body_conversion_3x= (1/0.438)  # Number of pixels to make 1000 um (equivalent to 1mm) under 3x magnification. I measured 1mm e.g., length from 60-70 
Eye_conversion_8x= (1/1.869)  # Number of pixels to make 1000 um (eqiuvelent to 1mm) under 8x magnification. I measured 1mm e.g., length from 60-70

'''
Notes: 
    eye image has dimentsions of W:2592 and H:1944
    body image has dimensions of W:1600 and H:1200 
'''

# magnification_difference_in_pixels= 2592/1600  # this is 1.62)


# # # Create table with body eye ratios 
# # # Control -----
# Control_body_eye_ratios_Left= ( Control_Left_Body_Length / Control_Left_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
# Control_body_eye_ratios_Left.columns=['Body Length/ Eye Diameter Ratio']

# Control_body_eye_ratios_Right= ( Control_Right_Body_Length / Control_Right_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
# Control_body_eye_ratios_Right.columns=['Body Length/ Eye Diameter Ratio']

# Control_mean_eye_lengths = np.average([Control_Left_eye_major_axis_length, Control_Right_eye_major_axis_length], axis=0)

# Control_mean_body_lengths = np.average([Control_Left_Body_Length, Control_Left_Body_Length], axis=0)

# Control_body_eye_ratios= ( Control_mean_body_lengths / Control_mean_eye_lengths  * (8/3) ) * magnification_difference_in_pixels

# Control_eye_body_ratio= 1 / Control_body_eye_ratios

# Control_mean_head_body_ratio= np.average([ ((Control_Left_Head_Length * Body_conversion_3x) / (Control_Left_Body_Length * Body_conversion_3x)), 
#                                             ((Control_Right_Head_Length * Body_conversion_3x) / (Control_Right_Body_Length * Body_conversion_3x)) ], axis=0)


# # # Injected -----
# Injected_body_eye_ratios_Left= ( Injected_Left_Body_Length / Injected_Left_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
# Injected_body_eye_ratios_Left.columns=['Body Length/ Eye Diameter Ratio']

# Injected_body_eye_ratios_Right= ( Injected_Right_Body_Length / Injected_Right_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
# Injected_body_eye_ratios_Right.columns=['Body Length/ Eye Diameter Ratio']

# Injected_mean_eye_lengths = np.average([Injected_Left_eye_major_axis_length, Injected_Right_eye_major_axis_length], axis=0)

# Injected_mean_body_lengths = np.average([Injected_Left_Body_Length, Injected_Left_Body_Length], axis=0)

# Injected_body_eye_ratios= ( Injected_mean_body_lengths / Injected_mean_eye_lengths  * (8/3) ) * magnification_difference_in_pixels

# Injected_eye_body_ratio = 1 / Injected_body_eye_ratios

# Injected_mean_head_body_ratio= np.average([ ((Injected_Left_Head_Length * Body_conversion_3x) / (Injected_Left_Body_Length * Body_conversion_3x)), 
#                                             ((Injected_Right_Head_Length * Body_conversion_3x) / (Injected_Right_Body_Length * Body_conversion_3x)) ], axis=0)


#%% # Do Stats for Eye Length
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_All_Eye_Length_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Eye Lengths: Control and Injected
s1= Combined_Control_eye_major_axis_length * Eye_conversion_8x
s2= Combined_Injected_eye_major_axis_length * Eye_conversion_8x
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('All Eye Length: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(Combined_Control_eye_major_axis_length) - np.mean(Combined_Injected_eye_major_axis_length)) / np.mean(Combined_Control_eye_major_axis_length)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change)  )

#------------------------
reportFile.close()


#%% # Box Plot with Scatter all eye Lengths 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s1= pd.concat([Control_Left_eye_major_axis_length, Control_Right_eye_major_axis_length], axis=0)
s1= pd.DataFrame(s1)
s1.columns=['Control']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= pd.concat([Injected_Left_eye_major_axis_length, Injected_Right_eye_major_axis_length], axis=0)
s2= pd.DataFrame(s2)
s2.columns=['Injected']
s2 = s2.set_index(np.arange(0,len(s2)))

# Create dataframe for plots
data_df1= s1.join(s2, how='right')
data_df1= data_df1 * Eye_conversion_8x

#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(1)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', ]
markers= ['.', '^']
sizes= [45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Eye Length')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)

    
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/All_Eye_Lengths-Boxplot_with_scatter', dpi=900)  

plt.show()    


#%% # Prep Eye Area Data For stats and Figures
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

''' 
# I got  the single_pixel_width_in_um= 0.5350455 value from Fiji. 
Fiji >image >properties (1 pixel is 0.0104167 inches therfore: 1pixel = 0.000406902 mm and 1 pixel  = 0.406902344 um)
You need to first set the scale to get the correct measurment. 
Follow instructions from https://www.otago.ac.nz/__data/assets/pdf_file/0028/301789/spatial-calibration-of-an-image-684707.pdf
'''

single_pixel_width_in_um= 0.5350455
squared_single_pixel_width_in_um= single_pixel_width_in_um * single_pixel_width_in_um
 
''' multiply the eye area pixels (obtained from skimage props regions) 
by squared_single_pixel_width_in_um to get the um^2 values. See link: 
    https://forum.image.sc/t/regionprops-area-conversion-to-micron-unit-of-measure/84779/2'''

##  Control
Combined_Control_eye_area= Combined_Control_eye_area * squared_single_pixel_width_in_um 

#Injected
Combined_Injected_eye_area= Combined_Injected_eye_area * squared_single_pixel_width_in_um 



#%% # Do Stats for All Eye Areas
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_All_Eye_Area_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Average Eye Area: Control and Injected
s1= Combined_Control_eye_area 
s2= Combined_Injected_eye_area
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('All Eye Area: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(Combined_Control_eye_area) - np.mean(Combined_Injected_eye_area)) / np.mean(Combined_Control_eye_area)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change)  )

#------------------------
reportFile.close()


#%% # Box Plot with Scatter all Eye Area 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s1= pd.concat([Control_Left_eye_area, Control_Right_eye_area], axis=0)
s1= pd.DataFrame(s1)
s1= s1 * squared_single_pixel_width_in_um
s1.columns=['Control']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= pd.concat([Injected_Left_eye_area, Injected_Right_eye_area], axis=0)
s2= pd.DataFrame(s2)
s2= s2 * squared_single_pixel_width_in_um
s2.columns=['Injected']
s2 = s2.set_index(np.arange(0,len(s2)))


# Create dataframe for plots
data_df1= s1.join(s2, how='right')


#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(2)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', ]
markers= ['.', '^']
sizes= [45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Eye Area')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)

    
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/All_Eye_Areas-Boxplot_with_scatter', dpi=900)  

plt.show()    

#%% # Do Stats for Average Body Length (Average Body Length per Fish)
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Control_vs_Injected_Body_Length.txt')
reportFile = open(reportFilename, 'w')


#-----------
# Average Body Lengths: Control and Injected
# Control
s1= pd.concat([Control_Left_Body_Length, Control_Left_Body_Length ], axis=1)
s1= s1 * Body_conversion_3x    
s1= np.mean(s1, axis=1)

# Injected
s2= pd.concat([Injected_Left_Body_Length, Injected_Left_Body_Length ], axis=1)
s2= s2 * Body_conversion_3x    
s2= np.mean(s2, axis=1)

reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('All Eye Area: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change)  )

#------------------------
reportFile.close()


#%% # Box Plot with Scatter Average Body Lengths 88888888888
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Average Body Lengths: Control and Injected
s1= pd.concat([Control_Left_Body_Length, Control_Left_Body_Length ], axis=1)
s1= s1 * Body_conversion_3x    
s1= np.mean(s1, axis=1)
s1= pd.DataFrame([s1]).T
s1.columns=['Control']
s1 = s1.set_index(np.arange(0,len(s1)))


s2= pd.concat([Injected_Left_Body_Length, Injected_Left_Body_Length ], axis=1)
s2= s2 * Body_conversion_3x    
s2= np.mean(s2, axis=1)
s2= pd.DataFrame([s2]).T
s2.columns=['Injected']
s2 = s2.set_index(np.arange(0,len(s2)))

# Create dataframe for plots
data_df1= s1.join(s2, how='right')


#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(3)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', ]
markers= ['.', '^']
sizes= [45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Body Length')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)

    
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Body_Lengths-Boxplot_with_scatter', dpi=900)  

plt.show()    

#%% # Do Stats for Eccentricity
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Control_vs_Injected_Eccentricity.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Eccentricity all Control fish eyes vs all Injected fish eyes
s1= Combined_Control_eye_eccentricity
s2= Combined_Injected_eye_eccentricity
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Eccentricity all Control fish eyes vs all Injected fish eyes' + '\n')  

# find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Control = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# paired t-test
t, p = sp.stats.ttest_ind(s1, s2)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
print ('U_statistic = ' + str(u) + '\n' + 'pValue ='  + str(p) )
line = "P-Value (UnInjected vs. Injected: " + str(p) + ' (MannWhiteney U-test)'
reportFile.write(line + '\n')


Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change)  )

#------------------------
reportFile.close()


#%% # Box Plot with Scatter for Eccentricity
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s1= Combined_Control_eye_eccentricity
s1=pd.DataFrame(s1)
s1.columns=['Control']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= Combined_Injected_eye_eccentricity
s2=pd.DataFrame(s2)
s2.columns=['Injected']
s2 = s2.set_index(np.arange(0,len(s2)))

# Create dataframe for plots
data_df1= s1.join(s2, how='right')

#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(4)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', ]
markers= ['.', '^']
sizes= [45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Eye Eccentricity')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)

 
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/All_Eye_Eccentricity-Boxplot_with_scatter', dpi=900)  
      
plt.show()  


#%% # Do Stats for Eye to Body Ratio
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Control_vs_Injected_Eye_Body_Ratio.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Eye Body Ratio all Control fish eyes vs all Injected fish eyes

s1a= Combined_Control_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s1b= Combined_Control_Body_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once

s2a= Combined_Injected_eye_major_axis_length * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s2b= Combined_Injected_Body_Length           * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
  
s1= s1a/s1b
s2= s2a/s2b

reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Eye Body Ratio all Control fish eyes vs all Injected fish eyes' + '\n')  

# find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Control = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# paired t-test
t, p = sp.stats.ttest_ind(s1, s2)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
print ('U_statistic = ' + str(u) + '\n' + 'pValue ='  + str(p) )
line = "P-Value (UnInjected vs. Injected: " + str(p) + ' (MannWhiteney U-test)'
reportFile.write(line + '\n')


Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change)  )

#------------------------
reportFile.close()


#%% # Box Plot with Scatter for Eye Body Ratio
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s1a= Combined_Control_eye_major_axis_length * Eye_conversion_8x              #Finding average then average fo that is the same as finding the average once
s2a= Combined_Injected_eye_major_axis_length * Eye_conversion_8x            #Finding average then average fo that is the same as finding the average once
    
s1b= Combined_Control_Body_Length * Body_conversion_3x               #Finding average then average fo that is the same as finding the average once
s2b= Combined_Injected_Body_Length * Body_conversion_3x              #Finding average then average fo that is the same as finding the average once

# Control    
s1= s1a/s1b
s1=pd.DataFrame(s1)
s1.columns=['Control']
s1 = s1.set_index(np.arange(0,len(s1)))

# Injected
s2= s2a/s2b
s2=pd.DataFrame(s2)
s2.columns=['Injected']
s2 = s2.set_index(np.arange(0,len(s2)))

# Create dataframe for plots
data_df1= s1.join(s2, how='right')

#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(5)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', ]
markers= ['.', '^']
sizes= [45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Eye Body Ratio')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)

 
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/All_Eye_Body_Ratio-Boxplot_with_scatter', dpi=900)       

plt.show()  


#%% # Do Stats for Head to Body Ratio
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Control_vs_Injected_Head_Body_Ratio.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Head to Body Ratio: Control vs Injected

# Control 
# Head Length
s1a= np.average([Control_Left_Head_Length, Control_Right_Head_Length], axis=0)
s1a= s1a * Body_conversion_3x
# Body Length
s1b= np.average([Control_Left_Body_Length, Control_Right_Body_Length], axis=0)
s1b= s1b * Body_conversion_3x


# Injected 
# Head Length
s2a= np.average([Injected_Left_Head_Length, Injected_Right_Head_Length], axis=0)
s2a= s2a * Body_conversion_3x
# Body Length
s2b= np.average([Injected_Left_Body_Length, Injected_Right_Body_Length], axis=0)
s2b= s2b * Body_conversion_3x

s1= s1a/s1b
s2= s2a/s2b

reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Head to Body Ratio Averaged Control vs Injected ' + '\n')  

# find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Control = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# paired t-test
t, p = sp.stats.ttest_ind(s1, s2)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
print ('U_statistic = ' + str(u) + '\n' + 'pValue ='  + str(p) )
line = "P-Value (UnInjected vs. Injected: " + str(p) + ' (MannWhiteney U-test)'
reportFile.write(line + '\n')


Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change)  )

#------------------------
reportFile.close()


#%% # Box Plot with Scatter for Head to Body Ratio
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Control 
# Head Length
s1a= np.average([Control_Left_Head_Length, Control_Right_Head_Length], axis=0)
s1a= s1a * Body_conversion_3x
# Body Length
s1b= np.average([Control_Left_Body_Length, Control_Right_Body_Length], axis=0)
s1b= s1b * Body_conversion_3x


# Injected 
# Head Length
s2a= np.average([Injected_Left_Head_Length, Injected_Right_Head_Length], axis=0)
s2a= s2a * Body_conversion_3x
# Body Length
s2b= np.average([Injected_Left_Body_Length, Injected_Right_Body_Length], axis=0)
s2b= s2b * Body_conversion_3x

# Control
s1= s1a/s1b
s1=pd.DataFrame(s1)
s1.columns=['Control']
s1 = s1.set_index(np.arange(0,len(s1)))

# Injected
s2= s2a/s2b
s2=pd.DataFrame(s2)
s2.columns=['Injected']
s2 = s2.set_index(np.arange(0,len(s2)))

# Create dataframe for plots
data_df1= s1.join(s2, how='right')

#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(6)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', ]
markers= ['.', '^']
sizes= [45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Head Body Ratio')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)

 
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/All_Head_Body_Ratio-Boxplot_with_scatter', dpi=900)        

plt.show()  


#%% # Do Stats for Eye to Head Ratio
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Control_vs_Injected_Eye_Head_Ratio.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Eye to Head Ratio: Control vs Injected

# Control 
# Eye Length
s1a= Combined_Control_eye_major_axis_length
s1a= s1a * Eye_conversion_8x
# Head Length
s1b= Combined_Control_Head_Length
s1b= s1b * Body_conversion_3x

# Injected
# Eye Length 
s2a= Combined_Injected_eye_major_axis_length
s2a= s2a * Eye_conversion_8x
# Head Length
s2b= Combined_Injected_Head_Length
s2b= s2b * Body_conversion_3x

s1= s1a/s1b
s2= s2a/s2b

reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Eye to Head Ratio Control vs Injected ' + '\n')  

# find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Control = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# paired t-test
t, p = sp.stats.ttest_ind(s1, s2)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
print ('U_statistic = ' + str(u) + '\n' + 'pValue ='  + str(p) )
line = "P-Value (UnInjected vs. Injected: " + str(p) + ' (MannWhiteney U-test)'
reportFile.write(line + '\n')


Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change)  )

#------------------------
reportFile.close()

#%% # Box Plot with Scatter for Eye to Head Ratio
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Eye to Head Ratio: Control vs Injected
# Control 
# Eye Length
s1a= Combined_Control_eye_major_axis_length
s1a= s1a * Eye_conversion_8x
# Head Length
s1b= Combined_Control_Head_Length
s1b= s1b * Body_conversion_3x

# Injected
# Eye Length 
s2a= Combined_Injected_eye_major_axis_length
s2a= s2a * Eye_conversion_8x
# Head Length
s2b= Combined_Injected_Head_Length
s2b= s2b * Body_conversion_3x

# Control
s1= s1a/s1b
s1=pd.DataFrame(s1)
s1.columns=['Control']
s1 = s1.set_index(np.arange(0,len(s1)))

# Injected
s2= s2a/s2b
s2=pd.DataFrame(s2)
s2.columns=['Injected']
s2 = s2.set_index(np.arange(0,len(s2)))

# Create dataframe for plots
data_df1= s1.join(s2, how='right')

#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(7)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', ]
markers= ['.', '^']
sizes= [45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Eye Head Ratio')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)

 
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Eye_Head_Ratio', dpi=900)        

plt.show()  




#%% ## Extra Plots Regression Plot Eye Area and Eccentricity

# # Control
# Combined_Control_eye_eccentricity = Combined_Control_eye_eccentricity.reset_index(drop=True)
# Combined_Control_eye_major_axis_length = Combined_Control_eye_major_axis_length.reset_index(drop=True)

# # Mean_Combined_Control_eye_eccentricity= np.mean(Combined_Control_eye_eccentricity)
# # Mean_Combined_Control_eye_major_axis_length= np.mean(Combined_Control_eye_major_axis_length)

# # Normalised_Combined_Control_eye_eccentricity = Combined_Control_eye_eccentricity/ Mean_Combined_Control_eye_eccentricity
# # Normalised_Combined_Control_eye_major_axis_length = Combined_Control_eye_major_axis_length/ Mean_Combined_Control_eye_major_axis_length


# # Injected
# Combined_Injected_eye_eccentricity = Combined_Injected_eye_eccentricity.reset_index(drop=True)
# Combined_Injected_eye_major_axis_length = Combined_Injected_eye_major_axis_length.reset_index(drop=True)

# # Normalised_Combined_Injected_eye_eccentricity = Combined_Injected_eye_eccentricity/ Mean_Combined_Control_eye_eccentricity
# # Normalised_Combined_Injected_eye_major_axis_length = Combined_Injected_eye_major_axis_length/ Mean_Combined_Control_eye_major_axis_length


# data_df1a = pd.concat([Combined_Control_eye_eccentricity, Combined_Control_eye_major_axis_length], axis=1)
# data_df1a.columns= ['Control_Eye_Eccentricity', 'Control_Eye_Length' ]

# data_df1b = pd.concat([Combined_Injected_eye_eccentricity, Combined_Injected_eye_major_axis_length], axis=1)
# data_df1b.columns= ['Injected_Eye_Eccentricity', 'Injected_Eye_Length' ]


# data_df1= pd.concat([data_df1a, data_df1b])

# fig = plt.figure(figsize=(10, 7))
# sns.regplot(x=data_df1a.Control_Eye_Length, y=data_df1a.Control_Eye_Eccentricity, color='black', marker='.')
# sns.regplot(x=data_df1b.Injected_Eye_Length, y=data_df1b.Injected_Eye_Eccentricity, color='black', marker='^')

# # legend, title, and labels.

# plt.title('Relationship between Eye Area and Eye Eccentricity', size=24)
# plt.xlabel('Eye Length (um)', size=18)
# plt.ylabel(' Eye Eccentricity ', size=18)
# #plt.ylim(260, 325)
# #plt.ylim(0, 1)


# data_df1a.corr()

# data_df1b.corr()

# # https://towardsdatascience.com/simple-and-multiple-linear-regression-with-python-c9ab422ec29c
# # https://stackoverflow.com/questions/66433019/how-to-statistically-compare-the-intercept-and-slope-of-two-different-linear-reg
# # https://realpython.com/linear-regression-in-python/


#%% Notes: 
    # ?? To check this: If we want to use the average eye measurements then all we need to do is change the graphs as the stats would be the same, average of average is the same as finding the average of the data once
    # For eye Body measurements, if you dont convert to um and repeat stats on pixel lengths, you still get significance 

#%% Finish

#%%