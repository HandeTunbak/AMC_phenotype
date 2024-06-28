# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 18:01:10 2022

@author: hande
"""


''' This code is to analyse the legnths and width of fish not treated in PTU. '''

#%% Read if first time running this code!

''' for first time use you will need to install

> seaborn:        $ conda install seaborn                                          - click y when prompted


 '''

#%% # import libraries
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

# Location with subdirectories
Experiment_folder = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_TL/Testing_area'

#------------------------------
# # Control
Control_Left_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/Control_Left_eye_parameters.csv') 

Control_Right_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/Control_Right_eye_parameters.csv')  

Control_Left_Body_Length_csv= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/Control_Left_Body_Length.csv') 

Control_Right_Body_Length_csv= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/Control_Right_Body_Length.csv') 

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
Control_Left_Body_Length= Control_Left_Body_Length_csv['Fish Body Length']
Control_Right_Body_Length= Control_Right_Body_Length_csv['Fish Body Length']
Combined_Control_Body_Length= pd.concat([Control_Left_Body_Length, 
                                               Control_Right_Body_Length])


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

Injected_Left_Body_Length_csv= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/Injected_Left_Body_Length.csv') 

Injected_Right_Body_Length_csv= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/Injected_Right_Body_Length.csv') 


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
Injected_Left_Body_Length= Injected_Left_Body_Length_csv['Fish Body Length']
Injected_Right_Body_Length= Injected_Right_Body_Length_csv['Fish Body Length']
Combined_Injected_Body_Length= pd.concat([Injected_Left_Body_Length, 
                                               Injected_Right_Body_Length])


# #     Injected Eye Major Axis Length
Injected_Left_eye_major_axis_length= Injected_Left_eye_paramaters['major_axis_length']
Injected_Right_eye_major_axis_length= Injected_Right_eye_paramaters['major_axis_length']
Combined_Injected_eye_major_axis_length = pd.concat([Injected_Left_eye_major_axis_length, 
                                               Injected_Right_eye_major_axis_length])



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

#------------------------
# Eccentricity all Control fish eyes left vs right
s3= Control_Left_eye_eccentricity
s4= Control_Right_eye_eccentricity
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Eccentricity all Control fish eyes: Left vs Right' + '\n')  

# find means
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)
reportFile.write('Mean Control Left = ' + str(s3_mean) + '\n')
reportFile.write('Mean Control Right = '  + str(s4_mean) + '\n')

# paired t-test
t, p = sp.stats.ttest_rel(s3, s4)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.wilcoxon(s3, s4)
print ('U_statistic = ' + str(u) + '\n' + 'pValue ='  + str(p) )
line = "P-Value (UnInjected: Left vs Right: " + str(p) + ' (Wilcoxon signed rank)'
reportFile.write(line + '\n')

#------------------------
# Eccentricity all Injected fish eyes left vs right
s5= Injected_Left_eye_eccentricity
s6= Injected_Right_eye_eccentricity
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Eccentricity all Injected fish eyes: Left vs Right' + '\n')  

# find means
s5_mean = np.mean(s5)
s6_mean = np.mean(s6)
reportFile.write('Mean Injected Left = ' + str(s5_mean) + '\n')
reportFile.write('Mean Injected Right = '  + str(s6_mean) + '\n')

# paired t-test
t, p = sp.stats.ttest_rel(s5, s6)
reportFile.write('\n' + 'Paired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.wilcoxon(s5, s6)
print ('U_statistic = ' + str(u) + '\n' + 'pValue ='  + str(p) )
line = "P-Value (Injected: Left vs Right: " + str(p) + ' (Wilcoxon signed rank)'
reportFile.write(line + '\n')

#------------------------
reportFile.close()


#%% # Prep Data and Plot Figures for Eccentricity
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s1=pd.DataFrame(s1)
s1.columns=['Control']
s1 = s1.set_index(np.arange(0,len(s1)))

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


plt.figure(1)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.25)

palette = ['r', 'g', 'b', 'y']

for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.4, color=c)
    plt.title('Eccentricity')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)

 
# # Save figure
# plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Eccentricity', dpi=400)        



#%% # Do Stats for Eye Area
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Controls 

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Control_vs Injected_Eye_Area.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Eccentricity all Control fish eyes vs all Injected fish eyes
s7= Combined_Control_eye_area
s8= Combined_Injected_eye_area
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('eye area all Control fish eyes vs all Injected fish eyes' + '\n')  

# find means
s7_mean = np.mean(s7)
s8_mean = np.mean(s8)
reportFile.write('Mean Control = ' + str(s7_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s8_mean) + '\n')

# paired t-test
t, p = sp.stats.ttest_ind(s7, s8)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s7, s8)
print ('U_statistic = ' + str(u) + '\n' + 'pValue ='  + str(p) )
line = "P-Value (UnInjected vs. Injected: " + str(p) + ' (MannWhiteney U-test)'
reportFile.write(line + '\n')

#------------------------
# Eccentricity all Control fish eyes left vs right
s9= Control_Left_eye_area
s10= Control_Right_eye_area
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Area all Control fish eyes: Left vs Right' + '\n')  

# find means
s9_mean = np.mean(s9)
s10_mean = np.mean(s10)
reportFile.write('Mean Control Left = ' + str(s9_mean) + '\n')
reportFile.write('Mean Control Right = '  + str(s10_mean) + '\n')

# paired t-test
t, p = sp.stats.ttest_rel(s9, s10)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.wilcoxon(s9, s10)
print ('U_statistic = ' + str(u) + '\n' + 'pValue ='  + str(p) )
line = "P-Value (UnInjected: Left vs Right: " + str(p) + ' (Wilcoxon signed rank)'
reportFile.write(line + '\n')

#------------------------
# Eccentricity all Injected fish eyes left vs right
s11= Injected_Left_eye_area
s12= Injected_Right_eye_area
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Area all Injected fish eyes: Left vs Right' + '\n')  

# find means
s11_mean = np.mean(s11)
s12_mean = np.mean(s12)
reportFile.write('Mean Injected Left = ' + str(s11_mean) + '\n')
reportFile.write('Mean Injected Right = '  + str(s12_mean) + '\n')

# paired t-test
t, p = sp.stats.ttest_rel(s11, s12)
reportFile.write('\n' + 'Paired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.wilcoxon(s11, s12)
print ('U_statistic = ' + str(u) + '\n' + 'pValue ='  + str(p) )
line = "P-Value (Injected: Left vs Right: " + str(p) + ' (Wilcoxon signed rank)'
reportFile.write(line + '\n')

#------------------------


reportFile.close()


#%% Prep Data and Plot Figures for Eye Area
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s7=pd.DataFrame(s7)
s7.columns=['Control']
s7 = s7.set_index(np.arange(0,len(s7)))

s8=pd.DataFrame(s8)
s8.columns=['Injected']
s8 = s8.set_index(np.arange(0,len(s8)))

# Create dataframe for plots
data_df2= s7.join(s8, how='right')

#----

vals, names, xs = [],[],[]
for i, col in enumerate(data_df2.columns):
    vals.append(data_df2[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df2[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(2)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.25)

palette = ['r', 'g', 'b', 'y']

for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.4, color=c)
    plt.title('Eye area')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)


# # Save figure
# plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Eye_area', dpi=400)  


#%% # Do Stats for Body Length
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Controls 

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Control_vs_Injected_Body_Length.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Eccentricity all Control fish eyes vs all Injected fish eyes
s13= Combined_Control_Body_Length * Body_conversion_3x
s14= Combined_Injected_Body_Length * Body_conversion_3x
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Body Length all Control vs all Injected' + '\n')  

# find means
s13_mean = np.mean(s13)
s14_mean = np.mean(s14)
reportFile.write('Mean Control = ' + str(s13_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s14_mean) + '\n')

# paired t-test
t, p = sp.stats.ttest_ind(s13, s14)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s13, s14)
print ('U_statistic = ' + str(u) + '\n' + 'pValue ='  + str(p) )
line = "P-Value (UnInjected vs. Injected: " + str(p) + ' (MannWhiteney U-test)'
reportFile.write(line + '\n')

#------------------------
# Eccentricity all Control fish eyes left vs right
s15= Control_Left_Body_Length * Body_conversion_3x
s16= Control_Right_Body_Length * Body_conversion_3x
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Body Length Control: Left vs Right' + '\n')  

# find means
s15_mean = np.mean(s15)
s16_mean = np.mean(s16)
reportFile.write('Mean Control Left = ' + str(s15_mean) + '\n')
reportFile.write('Mean Control Right = '  + str(s16_mean) + '\n')

# paired t-test
t, p = sp.stats.ttest_rel(s15, s16)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.wilcoxon(s15, s16)
print ('U_statistic = ' + str(u) + '\n' + 'pValue ='  + str(p) )
line = "P-Value (UnInjected: Left vs Right: " + str(p) + ' (Wilcoxon signed rank)'
reportFile.write(line + '\n')

#------------------------
# Eccentricity all Injected fish eyes left vs right
s17= Injected_Left_Body_Length * Body_conversion_3x
s18= Injected_Right_Body_Length * Body_conversion_3x
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Body Length Injected: Left vs Right' + '\n')  

# find means
s17_mean = np.mean(s17)
s18_mean = np.mean(s18)
reportFile.write('Mean Injected Left = ' + str(s17_mean) + '\n')
reportFile.write('Mean Injected Right = '  + str(s18_mean) + '\n')

# paired t-test
t, p = sp.stats.ttest_rel(s17, s18)
reportFile.write('\n' + 'Paired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.wilcoxon(s17, s18)
print ('U_statistic = ' + str(u) + '\n' + 'pValue ='  + str(p) )
line = "P-Value (Injected: Left vs Right: " + str(p) + ' (Wilcoxon signed rank)'
reportFile.write(line + '\n')

#------------------------
reportFile.close()


#%% # Prep Data and Plot Figures for Body Length
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s13=pd.DataFrame(s13)
s13.columns=['Control']
s13 = s13.set_index(np.arange(0,len(s13)))

s14=pd.DataFrame(s14)
s14.columns=['Injected']
s14 = s14.set_index(np.arange(0,len(s14)))

# Create dataframe for plots
data_df3= s13.join(s14, how='right')

#----

vals, names, xs = [],[],[]
for i, col in enumerate(data_df3.columns):
    vals.append(data_df3[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df3[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(3)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.25)

palette = ['r', 'g', 'b', 'y']

for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.4, color=c)
    plt.title('Body Length')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)


# Save figure
# plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Body_Length', dpi=400)  


#%% # Prep Data and Plot Figures for Body Length - Left side only


s15=pd.DataFrame(s15)
s15.columns=['Control']
s15 = s15.set_index(np.arange(0,len(s15)))

s17=pd.DataFrame(s17)
s17.columns=['Injected']
s17 = s17.set_index(np.arange(0,len(s17)))

# Create dataframe for plots
data_df4= s15.join(s17, how='right')

#----

vals, names, xs = [],[],[]
for i, col in enumerate(data_df4.columns):
    vals.append(data_df4[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df4[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(4)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.25)

palette = ['r', 'g', 'b', 'y']

for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.4, color=c)
    plt.title('Body Length- left only')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)


# Save figure
# plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Body_Length-Left-only', dpi=400)  


#%% # Prep Data and Plot Figures for Body Length - Right side only

s16=pd.DataFrame(s16)
s16.columns=['Control']
s16 = s16.set_index(np.arange(0,len(s16)))

s18=pd.DataFrame(s18)
s18.columns=['Injected']
s18 = s18.set_index(np.arange(0,len(s18)))

# Create dataframe for plots
data_df4= s16.join(s18, how='right')

#----

vals, names, xs = [],[],[]
for i, col in enumerate(data_df4.columns):
    vals.append(data_df4[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df4[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(5)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.25)

palette = ['r', 'g', 'b', 'y']

for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.4, color=c)
    plt.title('Body Length- right only')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)


# # Save figure
# plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Body_Length-Right-only', dpi=400)  







#%% Prep Body Eye Major Ratio and Plot Figures for Body Eye Ratio
# #-----------------------------------------
# #-----------------------------------------

# # Conversions from pixels to 1mm length
Body_conversion_3x= 1/875 # Number of pixels to make 1mm under 3x magnification
Eye_conversion_8x= 1/2150 # Number of pixels to make 1mm under 8x magnification

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

Control_body_eye_ratios_fish_ids= pd.concat([Control_Left_eye_paramaters['fish_IDs'], pd.DataFrame(Control_body_eye_ratios)], axis=1)


# # Injected -----
Injected_body_eye_ratios_Left= ( Injected_Left_Body_Length / Injected_Left_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Injected_body_eye_ratios_Left.columns=['Body Length/ Eye Diameter Ratio']

Injected_body_eye_ratios_Right= ( Injected_Right_Body_Length / Injected_Right_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Injected_body_eye_ratios_Right.columns=['Body Length/ Eye Diameter Ratio']

Injected_mean_eye_lengths = np.average([Injected_Left_eye_major_axis_length, Injected_Right_eye_major_axis_length], axis=0)

Injected_mean_body_lengths = np.average([Injected_Left_Body_Length, Injected_Left_Body_Length], axis=0)

Injected_body_eye_ratios= ( Injected_mean_body_lengths / Injected_mean_eye_lengths  * (8/3) ) * magnification_difference_in_pixels



#-----
s19=pd.DataFrame(Control_body_eye_ratios)
s19.columns=['Control']
s19 = s19.set_index(np.arange(0,len(s19)))

s20=pd.DataFrame(Injected_body_eye_ratios)
s20.columns=['Injected']
s20 = s20.set_index(np.arange(0,len(s20)))

# Create dataframe for plots
data_df5= s19.join(s20, how='right')

#----

vals, names, xs = [],[],[]
for i, col in enumerate(data_df5.columns):
    vals.append(data_df5[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df5[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(6)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.25)

palette = ['r', 'g', 'b', 'y']

for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.4, color=c)
    plt.title('Average Body /Eye Ratio per Fish')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)


# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Body_Eye_Ratios_per_Fish', dpi=400)  


#-----
s19=pd.DataFrame(Control_mean_body_lengths)
s19.columns=['Control']
s19 = s19.set_index(np.arange(0,len(s19)))

s20=pd.DataFrame(Injected_mean_body_lengths)
s20.columns=['Injected']
s20 = s20.set_index(np.arange(0,len(s20)))

# Create dataframe for plots
data_df5= s19.join(s20, how='right')

#----

vals, names, xs = [],[],[]
for i, col in enumerate(data_df5.columns):
    vals.append(data_df5[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df5[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(7)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.25)

palette = ['r', 'g', 'b', 'y']

for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.4, color=c)
    plt.title('Mean Body Length: Control vs Injected')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)


# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Body_length_Control_vs_Injected', dpi=400)  


#-----
s19=pd.DataFrame(Control_mean_eye_lengths)
s19.columns=['Control']
s19 = s19.set_index(np.arange(0,len(s19)))

s20=pd.DataFrame(Injected_mean_eye_lengths)
s20.columns=['Injected']
s20 = s20.set_index(np.arange(0,len(s20)))

# Create dataframe for plots
data_df5= s19.join(s20, how='right')

#----

vals, names, xs = [],[],[]
for i, col in enumerate(data_df5.columns):
    vals.append(data_df5[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df5[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(8)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.25)

palette = ['r', 'g', 'b', 'y']

for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.4, color=c)
    plt.title('Mean Eye Length: Control vs Injected')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)


# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_length_Control_vs_Injected', dpi=400)  

labels=['Control', 'Injected']
Control= Control_mean_eye_lengths
Control_mean= Control.mean(axis=0)

Injected= Injected_mean_eye_lengths
Injected_mean= Injected.mean(axis=0)



y1 = (np.random.random(100) - 0.5).cumsum()
y2 = y1.reshape(-1, 10).mean(axis=1)

plt.bar(labels, (Control_mean, Injected_mean), width=0.25)

plt.plot(labels[0], Control_mean)
plt.plot(labels[1], Injected_mean)

plt.show()




 #Set the figure size
plt.figure()

    
Control= pd.DataFrame([Control_mean_eye_lengths])
Injected= pd.DataFrame([Injected_mean_eye_lengths])

data_df= pd.concat([Control, Injected]).T
data_df.columns= ['Control', 'Injected']
data_df_0= data_df.T
colour = ['red', 'blue']

x_data= (data_df.columns).tolist()




# plot a bar chart

#  Control
Control= Control_mean_eye_lengths
Control_mean= Control.mean(axis=0)
Control_mean= pd.DataFrame([    ('Control:Mean' , Control_mean) ])
sem_Control= pd.DataFrame([stats.sem(Control)])

# Injected
Injected= Injected_mean_eye_lengths
Injected_mean= Injected.mean(axis=0)
Injected_mean= pd.DataFrame([    ('Injected:Mean' , Injected_mean) ])
sem_Injected= pd.DataFrame([stats.sem(Injected)])


# Create DataFrame with the two condtion means
sem_df= pd.concat([sem_Control, sem_Injected], axis=0)
sem_df.columns=['Standard Error']

data_df= pd.concat([Control_mean , Injected_mean], axis=0)
data_df.columns=['Conditions', 'Averages']

data_df2= pd.concat([data_df, sem_Injected ], axis=1)

data_df2.columns=['Conditions', 'Averages' , 'Standard Error']



sem_bars= pd.concat()


# Plot Figure

plt.figure(9)
ax= plt.subplot()

color= ['r', 'g']

Control_df=pd.DataFrame([Control]).T
Control_df.columns=['Control']

Injected_df=pd.DataFrame([Injected]).T
Injected_df.columns=['Injected']



df2 = pd.DataFrame({'x': [1, 2], 'y1': Control_df, 'y2': Injected})












ax= sns.barplot( data=Control_df, estimator=np.mean, errorbar=('se'),  capsize=.2, color='blue')
ax1= sns.barplot( data=Injected_df, estimator=np.mean, errorbar=('se'),  capsize=.2, color='green')

x= ['Control', 'Injected']

ax1 = sns.barplot( x=data_df2['Conditions'], y= data_df2['Averages'], data=data_df2, estimator=np.mean, errorbar=('se'),  capsize=.2, color='lightblue')











sns.barplot( x=data_df2['Conditions'], y= data_df2['Averages'],  
            data=data_df, hue=data_df['Conditions'], palette=color) 

ax.spines[['right', 'top']].set_visible(False)

with plt.errorbar(x = data_df2['Conditions'], y = data_df2['Averages'],
            yerr=data_df2['Standard Error'], fmt='none', c= 'black', capsize = 10)



plt.figure()
ax= plt.subplot()


ax.bar( y='Control', = Control.mean(), data=Control,  errorbar=('se'),  capsize=.2, color='lightblue') 

    
    ax.bar( x= 'c', data=Control, estimator=np.mean, errorbar=('se'),  capsize=.2, color='lightblue') #hue='class', palette='Blues' ,
ax.spines[['right', 'top']].set_visible(False)



#%% ## Finish    
    
    
    
        