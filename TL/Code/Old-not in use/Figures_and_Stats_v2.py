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
Experiment_folder = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_TL/Testing_area'

#------------------------------
# # Control
Control_Left_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/Control_Left_eye_parameters.csv') 

Control_Right_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/Control_Right_eye_parameters.csv')  

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
                                               '/Injected' + '/Injected_Left_eye_parameters.csv') 

Injected_Right_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/Injected_Right_eye_parameters.csv')  

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

magnification_difference_in_pixels= 2592/1600  # this is 1.62)


# # Create table with body eye ratios 
# # Control -----
Control_body_eye_ratios_Left= ( Control_Left_Body_Length / Control_Left_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Control_body_eye_ratios_Left.columns=['Body Length/ Eye Diameter Ratio']

Control_body_eye_ratios_Right= ( Control_Right_Body_Length / Control_Right_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Control_body_eye_ratios_Right.columns=['Body Length/ Eye Diameter Ratio']

Control_mean_eye_lengths = np.average([Control_Left_eye_major_axis_length, Control_Right_eye_major_axis_length], axis=0)

Control_mean_body_lengths = np.average([Control_Left_Body_Length, Control_Left_Body_Length], axis=0)

Control_body_eye_ratios= ( Control_mean_body_lengths / Control_mean_eye_lengths  * (8/3) ) * magnification_difference_in_pixels

Control_eye_body_ratio= 1 / Control_body_eye_ratios

Control_mean_head_body_ratio= np.average([ ((Control_Left_Head_Length * Body_conversion_3x) / (Control_Left_Body_Length * Body_conversion_3x)), 
                                            ((Control_Right_Head_Length * Body_conversion_3x) / (Control_Right_Body_Length * Body_conversion_3x)) ], axis=0)


# # Injected -----
Injected_body_eye_ratios_Left= ( Injected_Left_Body_Length / Injected_Left_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Injected_body_eye_ratios_Left.columns=['Body Length/ Eye Diameter Ratio']

Injected_body_eye_ratios_Right= ( Injected_Right_Body_Length / Injected_Right_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Injected_body_eye_ratios_Right.columns=['Body Length/ Eye Diameter Ratio']

Injected_mean_eye_lengths = np.average([Injected_Left_eye_major_axis_length, Injected_Right_eye_major_axis_length], axis=0)

Injected_mean_body_lengths = np.average([Injected_Left_Body_Length, Injected_Left_Body_Length], axis=0)

Injected_body_eye_ratios= ( Injected_mean_body_lengths / Injected_mean_eye_lengths  * (8/3) ) * magnification_difference_in_pixels

Injected_eye_body_ratio = 1 / Injected_body_eye_ratios

Injected_mean_head_body_ratio= np.average([ ((Injected_Left_Head_Length * Body_conversion_3x) / (Injected_Left_Body_Length * Body_conversion_3x)), 
                                            ((Injected_Right_Head_Length * Body_conversion_3x) / (Injected_Right_Body_Length * Body_conversion_3x)) ], axis=0)


#%% # Do Stats for Average Eye Length
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Average_Eye_Length_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Average Eye Lengths: Control and Injected
s1= Control_mean_eye_lengths
s2= Injected_mean_eye_lengths
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Average Eye Length: Control vs Injected' + '\n' + '\n')  

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

Percentage_change = ((np.mean(Control_mean_eye_lengths) - np.mean(Injected_mean_eye_lengths)) / np.mean(Control_mean_eye_lengths)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change)  )

#------------------------
reportFile.close()


#%% # Plot Average Eye Length
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s1= Control_mean_eye_lengths
s1= pd.DataFrame([s1])

s2= Injected_mean_eye_lengths
s2= pd.DataFrame([s2])

df= pd.concat([ s1, s2 ], axis=0).T
df.columns=['Control', 'Injected']

df= df * Eye_conversion_8x # Number of pixels to make 1 um under 8x magnification


#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot( data=df, estimator='mean', errorbar=(('se', 2)), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Axial Length (µm)', fontsize=10 )
plt.title(' Average Eye Length: Control vs Injected')
plt.ylim(260, 325)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Length_Control_vs_Injected', dpi=900)  

plt.show()

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot( data=df, estimator='mean', errorbar=(('se', 2)), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Axial Length (µm)', fontsize=10 )
plt.title(' Average Eye Length: Control vs Injected')
# plt.ylim(260, 325)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Length_Control_vs_Injected_Default', dpi=900)  

plt.show()

#%% # Plot all eye Lengths bar with scatter and jitter


s1= pd.concat([Control_Left_eye_major_axis_length, Control_Right_eye_major_axis_length], axis=0)
s1= pd.DataFrame(s1)
s1.columns=['Control']
s1= s1.reindex()

s2= pd.concat([Injected_Left_eye_major_axis_length, Injected_Right_eye_major_axis_length], axis=0)
s2= pd.DataFrame(s2)
s2.columns=['Injected']
s2= s2.reindex()


df1= pd.DataFrame({
    'Condition': ['Control', 'Injected'], 
    'Lengths': [ (s1['Control'] * Eye_conversion_8x), 
                 (s2['Injected'] * Eye_conversion_8x)  ] })


vals, names, xs = [],[],[]
for i, col in enumerate(df1.columns):
    vals.append(df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot( data=df, estimator='mean', errorbar=(('se', 2)), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Axial Length (µm)', fontsize=10 )
plt.title(' Average Eye Length: Control vs Injected')
plt.ylim(260, 325)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Length_Control_vs_Injected_Default', dpi=900)  

plt.show()

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot( data=df, estimator='mean', errorbar=(('se', 2)), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Axial Length (µm)', fontsize=10 )
plt.title(' Average Eye Length: Control vs Injected')
plt.ylim(260, 325)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Length_Control_vs_Injected_Default', dpi=900)  

plt.show()


#%% # Box Plot with Scatter all eye Lengths 

#------------------
# # Plot Boxplot with Scatter plot

s1= pd.concat([Control_Left_eye_major_axis_length, Control_Right_eye_major_axis_length], axis=0)
s1= pd.DataFrame(s1)
s1.columns=['Control']
s1= s1.reindex()

s2= pd.concat([Injected_Left_eye_major_axis_length, Injected_Right_eye_major_axis_length], axis=0)
s2= pd.DataFrame(s2)
s2.columns=['Injected']
s2= s2.reindex()


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
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.25)

palette = ['black', 'black', ]
markers= ['.', '^']
sizes= [35, 25]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Eye Length')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)

    
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/All_Eye_Lengths-Boxplot_with_scatter', dpi=900)  

plt.show()    

#%% # Do Stats for Average Body Length
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Average_Body_Length_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Average Body Lengths: Control and Injected
s3= Control_mean_body_lengths * Body_conversion_3x 
s4= Injected_mean_body_lengths * Body_conversion_3x
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Average Body Length: Control vs Injected' + '\n' + '\n')  

# Find means
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)
reportFile.write('Mean Uninjected = ' + str(s3_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s4_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s3, s4)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s3, s4)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(Control_mean_body_lengths) - np.mean(Injected_mean_body_lengths)) / np.mean(Control_mean_body_lengths)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change)  )

#------------------------
reportFile.close()


#%% # Plot Average Body Length
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s3= Control_mean_body_lengths
s3= pd.DataFrame([s3])

s4= Injected_mean_body_lengths
s4= pd.DataFrame([s4])

df2= pd.concat([ s3, s4 ], axis=0).T
df2.columns=['Control', 'Injected']

df2= df2 * Body_conversion_3x # Number of pixels to make 1mm under 8x magnification


#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(  data=df2, estimator='mean', errorbar=(('se', 2)), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Body Length (µm)', fontsize=10 )
plt.title(' Average Body Length: Control vs Injected')
plt.ylim(2600, 3600)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Body_Length_Control_vs_Injected', dpi=900)  

plt.show()

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(  data=df2, estimator='mean', errorbar=(('se', 2)), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Body Length (µm)', fontsize=10 )
plt.title(' Average Body Length: Control vs Injected')
# plt.ylim(2600, 3600)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Body_Length_Control_vs_Injected_Default', dpi=900)  

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
# find average across fish (find average of left an right measurements)
Control_avg_eye_area= np.average([Combined_Control_eye_area], axis=0)
Control_avg_eye_area= Control_avg_eye_area * squared_single_pixel_width_in_um  

#Injected
Injected_avg_eye_area= np.average([Combined_Injected_eye_area], axis=0)
Injected_avg_eye_area= Injected_avg_eye_area * squared_single_pixel_width_in_um 


#%% # Do Stats for Average Eye Area
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Average_Eye_Area_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Average Eye Area: Control and Injected
s5= Control_avg_eye_area
s6= Injected_avg_eye_area
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Average Eye Area: Control vs Injected' + '\n' + '\n')  

# Find means
s5_mean = np.mean(s5)
s6_mean = np.mean(s6)
reportFile.write('Mean Uninjected = ' + str(s5_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s6_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s5, s6)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s5, s6)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(Control_avg_eye_area) - np.mean(Injected_avg_eye_area)) / np.mean(Control_avg_eye_area)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change)  )

#------------------------
reportFile.close()


#%% # Plot Average Eye Area
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s5= Control_avg_eye_area
s5= pd.DataFrame([s5])

s6= Injected_avg_eye_area
s6= pd.DataFrame([s6])

df3= pd.concat([ s5, s6 ], axis=0).T
df3.columns=['Control', 'Injected']


#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(  data=df3, estimator='mean', errorbar=(('se', 2)), capsize=.2, errwidth=2.1, color='lightblue', width=.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Eye Area (µm^2)', fontsize=10 )
plt.title(' Average Eye Area: Control vs Injected')
plt.ylim(50000, 70000)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Area_Control_vs_Injected', dpi=900)  

plt.show()

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(  data=df3, estimator='mean', errorbar=(('se', 2)), capsize=.2, errwidth=2.1, color='lightblue', width=.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Eye Area (µm^2)', fontsize=10 )
plt.title(' Average Eye Area: Control vs Injected')
# plt.ylim(50000, 70000)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Area_Control_vs_Injected_Default', dpi=900)  

plt.show()


#%% # Do Stats for Average Eye Body Ratio
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Average_Eye_Body_Ratio_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Average Eye Body Ratio: Control and Injected
s7= Control_eye_body_ratio
s8= Injected_eye_body_ratio
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Average Eye Body Ratio: Control vs Injected' + '\n' + '\n')  

# Find means
s7_mean = np.mean(s7)
s8_mean = np.mean(s8)
reportFile.write('Mean Uninjected = ' + str(s7_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s8_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s7, s8)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s7, s8)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(Control_eye_body_ratio) - np.mean(Injected_eye_body_ratio)) / np.mean(Control_eye_body_ratio)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change)  )

#------------------------
reportFile.close()


#%% # Plot Average Eye Body Ratio 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s7= Control_eye_body_ratio
s7= pd.DataFrame([s7])

s8= Injected_eye_body_ratio
s8= pd.DataFrame([s8])

df4= pd.concat([ s7, s8 ], axis=0).T
df4.columns=['Control', 'Injected']

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(data=df4, estimator='mean', errorbar=(('se', 2)), capsize=.2, errwidth=2.1, color='lightblue', width=.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Eye/ Body Ratio', fontsize=10 )
plt.title(' Average Eye Body Ratio: Control vs Injected')
plt.ylim(0.045, 0.105)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Body_Ratio_Control_vs_Injected', dpi=900)  

plt.show()

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(data=df4, estimator='mean', errorbar=(('se', 2)), capsize=.2, errwidth=2.1, color='lightblue', width=.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Eye/ Body Ratio', fontsize=10 )
plt.title(' Average Eye Body Ratio: Control vs Injected')
# plt.ylim(0.045, 0.105)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Body_Ratio_Control_vs_Injected_Default', dpi=900)  

plt.show()

#%% # Not in use --> Double checking the eye/body ratio to be extra sure 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# x= Control_mean_body_lengths * Body_conversion_3x 
# y= Control_mean_eye_lengths * Eye_conversion_8x
# control_ratio= y/x
# control_ratios= np.average( pd.DataFrame([control_ratio]), axis=1)

# x= Injected_mean_body_lengths * Body_conversion_3x 
# y= Injected_mean_eye_lengths * Eye_conversion_8x
# injected_ratio= y/x
# injected_ratios= np.average( pd.DataFrame([injected_ratio]), axis=1)


# s9= control_ratio
# s10= injected_ratio

# t, p1 = sp.stats.ttest_ind(s9, s10)
# u, p2 = sp.stats.mannwhitneyu(s9, s10)


#%% # List of used websites

'''
Used websites:
    https://www.otago.ac.nz/__data/assets/pdf_file/0028/301789/spatial-calibration-of-an-image-684707.pdf
    https://monashmicroimaging.github.io/gitbook-fiji-basics/part_4_scales_and_sizing.html
    https://forum.image.sc/t/regionprops-area-conversion-to-micron-unit-of-measure/84779/2
    https://imagej.net/ij/docs/menus/analyze.html#:~:text=Area%20%2D%20Area%20of%20selection%20in,gray%20value%20within%20the%20selection.
    https://scikit-image.org/docs/stable/api/skimage.measure.html
'''


#%% # Finish




#%% Do stats for head body ratio

# Control 

Control_mean_Head_length= np.average([ Control_Left_Head_Length, Control_Right_Head_Length], axis=0)
Control_mean_Tail_length= np.average([ Control_Left_Tail_Length, Control_Right_Tail_Length], axis=0)
Control_mean_head_body_ratio= np.average([ (Control_mean_Head_length/Control_mean_Tail_length ) ], axis=0)


Injected_mean_Head_length= np.average([ Injected_Left_Head_Length, Injected_Right_Head_Length], axis=0)
Injected_mean_Tail_length= np.average([ Injected_Left_Tail_Length, Injected_Right_Tail_Length], axis=0)
Injected_mean_head_body_ratio= np.average([ (Injected_mean_Head_length/Injected_mean_Tail_length ) ], axis=0)



s1= Control_mean_head_body_ratio
s2= Injected_mean_head_body_ratio

t, p = sp.stats.ttest_ind(s1, s2)


Control_tail_eye_ratios= ( Control_mean_Tail_length / ((Control_mean_eye_lengths)  * (8/3))  * magnification_difference_in_pixels)
Control_eye_tail_ratios = 1/Control_tail_eye_ratios

Injected_tail_eye_ratios= ( Injected_mean_Tail_length / ((Injected_mean_eye_lengths)  * (8/3))  * magnification_difference_in_pixels)
Injected_eye_tail_ratios= 1/Injected_tail_eye_ratios

t, p = sp.stats.ttest_ind(Control_eye_tail_ratios, Injected_eye_tail_ratios)

np.mean(Control_eye_tail_ratios)
np.mean(Injected_eye_tail_ratios)


