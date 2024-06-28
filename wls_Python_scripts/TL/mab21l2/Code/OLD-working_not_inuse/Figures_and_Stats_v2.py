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
Experiment_folder = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area'

#------------------------------
# # Control MAB sibling
Control_MAB_siblings_Left_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/MAB_siblings' + '/MAB_siblings_Left_Eye_Parameters.csv') 

Control_MAB_siblings_Right_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/MAB_siblings' + '/MAB_siblings_Right_Eye_Parameters.csv')  

Control_MAB_siblings_Left_Body_Length_csv= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/MAB_siblings' + '/MAB_siblings_Left_Body_Length.csv') 

Control_MAB_siblings_Right_Body_Length_csv= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/MAB_siblings' + '/MAB_siblings_Right_Body_Length.csv') 


# #     Control MAB sibling Eccentricity
Control_MAB_siblings_Left_eye_eccentricity= Control_MAB_siblings_Left_eye_paramaters['eccentricity']
Control_MAB_siblings_Right_eye_eccentricity= Control_MAB_siblings_Right_eye_paramaters['eccentricity']
Combined_Control_MAB_siblings_eye_eccentricity = pd.concat([Control_MAB_siblings_Left_eye_eccentricity, 
                                               Control_MAB_siblings_Right_eye_eccentricity])

# #     Control MAB sibling Eye Area
Control_MAB_siblings_Left_eye_area= Control_MAB_siblings_Left_eye_paramaters['area']
Control_MAB_siblings_Right_eye_area= Control_MAB_siblings_Right_eye_paramaters['area']
Combined_Control_MAB_siblings_eye_area = pd.concat([Control_MAB_siblings_Left_eye_area, 
                                               Control_MAB_siblings_Right_eye_area])

# #     Control MAB sibling Body Length
Control_MAB_siblings_Left_Body_Length= Control_MAB_siblings_Left_Body_Length_csv['Fish Body Length']
Control_MAB_siblings_Right_Body_Length= Control_MAB_siblings_Right_Body_Length_csv['Fish Body Length']
Combined_Control_MAB_siblings_Body_Length= pd.concat([Control_MAB_siblings_Left_Body_Length, 
                                               Control_MAB_siblings_Right_Body_Length])

# #     Control MAB sibling Eye Major Axis Length
Control_MAB_siblings_Left_eye_major_axis_length= Control_MAB_siblings_Left_eye_paramaters['major_axis_length']
Control_MAB_siblings_Right_eye_major_axis_length= Control_MAB_siblings_Right_eye_paramaters['major_axis_length']
Combined_Control_MAB_siblings_eye_major_axis_length = pd.concat([Control_MAB_siblings_Left_eye_major_axis_length, 
                                               Control_MAB_siblings_Right_eye_major_axis_length])

##--------------------------------------
# # Control MAB phenos
Control_MAB_phenos_Left_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/MAB_phenos' + '/MAB_phenos_Left_Eye_Parameters.csv') 

Control_MAB_phenos_Right_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/MAB_phenos' + '/MAB_phenos_Right_Eye_Parameters.csv')  

Control_MAB_phenos_Left_Body_Length_csv= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/MAB_phenos' + '/MAB_phenos_Left_Body_Length.csv') 

Control_MAB_phenos_Right_Body_Length_csv= pd.read_csv(Experiment_folder + 
                                               '/Control' + '/MAB_phenos' + '/MAB_phenos_Right_Body_Length.csv') 


# #     Control MAB phenos Eccentricity
Control_MAB_phenos_Left_eye_eccentricity= Control_MAB_phenos_Left_eye_paramaters['eccentricity']
Control_MAB_phenos_Right_eye_eccentricity= Control_MAB_phenos_Right_eye_paramaters['eccentricity']
Combined_Control_MAB_phenos_eye_eccentricity = pd.concat([Control_MAB_phenos_Left_eye_eccentricity, 
                                               Control_MAB_phenos_Right_eye_eccentricity])

# #     Control MAB phenos Eye Area
Control_MAB_phenos_Left_eye_area= Control_MAB_phenos_Left_eye_paramaters['area']
Control_MAB_phenos_Right_eye_area= Control_MAB_phenos_Right_eye_paramaters['area']
Combined_Control_MAB_phenos_eye_area = pd.concat([Control_MAB_phenos_Left_eye_area, 
                                               Control_MAB_phenos_Right_eye_area])

# #     Control MAB phenos Body Length
Control_MAB_phenos_Left_Body_Length= Control_MAB_phenos_Left_Body_Length_csv['Fish Body Length']
Control_MAB_phenos_Right_Body_Length= Control_MAB_phenos_Right_Body_Length_csv['Fish Body Length']
Combined_Control_MAB_phenos_Body_Length= pd.concat([Control_MAB_phenos_Left_Body_Length, 
                                               Control_MAB_phenos_Right_Body_Length])

# #     Control MAB phenos Eye Major Axis Length
Control_MAB_phenos_Left_eye_major_axis_length= Control_MAB_phenos_Left_eye_paramaters['major_axis_length']
Control_MAB_phenos_Right_eye_major_axis_length= Control_MAB_phenos_Right_eye_paramaters['major_axis_length']
Combined_Control_MAB_phenos_eye_major_axis_length = pd.concat([Control_MAB_phenos_Left_eye_major_axis_length, 
                                               Control_MAB_phenos_Right_eye_major_axis_length])

##--------------------------------------
# # Injected MAB sibling
Injected_MAB_siblings_Left_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/MAB_siblings' + '/MAB_siblings_Left_Eye_Parameters.csv') 

Injected_MAB_siblings_Right_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/MAB_siblings' + '/MAB_siblings_Right_Eye_Parameters.csv')  

Injected_MAB_siblings_Left_Body_Length_csv= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/MAB_siblings' + '/MAB_siblings_Left_Body_Length.csv') 

Injected_MAB_siblings_Right_Body_Length_csv= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/MAB_siblings' + '/MAB_siblings_Right_Body_Length.csv') 


# #     Injected MAB sibling Eccentricity
Injected_MAB_siblings_Left_eye_eccentricity= Injected_MAB_siblings_Left_eye_paramaters['eccentricity']
Injected_MAB_siblings_Right_eye_eccentricity= Injected_MAB_siblings_Right_eye_paramaters['eccentricity']
Combined_Injected_MAB_siblings_eye_eccentricity = pd.concat([Injected_MAB_siblings_Left_eye_eccentricity, 
                                               Injected_MAB_siblings_Right_eye_eccentricity])

# #     Injected MAB sibling Eye Area
Injected_MAB_siblings_Left_eye_area= Injected_MAB_siblings_Left_eye_paramaters['area']
Injected_MAB_siblings_Right_eye_area= Injected_MAB_siblings_Right_eye_paramaters['area']
Combined_Injected_MAB_siblings_eye_area = pd.concat([Injected_MAB_siblings_Left_eye_area, 
                                               Injected_MAB_siblings_Right_eye_area])

# #     Injected MAB sibling Body Length
Injected_MAB_siblings_Left_Body_Length= Injected_MAB_siblings_Left_Body_Length_csv['Fish Body Length']
Injected_MAB_siblings_Right_Body_Length= Injected_MAB_siblings_Right_Body_Length_csv['Fish Body Length']
Combined_Injected_MAB_siblings_Body_Length= pd.concat([Injected_MAB_siblings_Left_Body_Length, 
                                               Injected_MAB_siblings_Right_Body_Length])

# #     Injected MAB sibling Eye Major Axis Length
Injected_MAB_siblings_Left_eye_major_axis_length= Injected_MAB_siblings_Left_eye_paramaters['major_axis_length']
Injected_MAB_siblings_Right_eye_major_axis_length= Injected_MAB_siblings_Right_eye_paramaters['major_axis_length']
Combined_Injected_MAB_siblings_eye_major_axis_length = pd.concat([Injected_MAB_siblings_Left_eye_major_axis_length, 
                                               Injected_MAB_siblings_Right_eye_major_axis_length])

##--------------------------------------
# # Injected MAB phenos
Injected_MAB_phenos_Left_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/MAB_phenos' + '/MAB_phenos_Left_Eye_Parameters.csv') 

Injected_MAB_phenos_Right_eye_paramaters= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/MAB_phenos' + '/MAB_phenos_Right_Eye_Parameters.csv')  

Injected_MAB_phenos_Left_Body_Length_csv= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/MAB_phenos' + '/MAB_phenos_Left_Body_Length.csv') 

Injected_MAB_phenos_Right_Body_Length_csv= pd.read_csv(Experiment_folder + 
                                               '/Injected' + '/MAB_phenos' + '/MAB_phenos_Right_Body_Length.csv') 


# #     Injected MAB phenos Eccentricity
Injected_MAB_phenos_Left_eye_eccentricity= Injected_MAB_phenos_Left_eye_paramaters['eccentricity']
Injected_MAB_phenos_Right_eye_eccentricity= Injected_MAB_phenos_Right_eye_paramaters['eccentricity']
Combined_Injected_MAB_phenos_eye_eccentricity = pd.concat([Injected_MAB_phenos_Left_eye_eccentricity, 
                                               Injected_MAB_phenos_Right_eye_eccentricity])


# #     Injected MAB phenos Body Length
Injected_MAB_phenos_Left_Body_Length= Injected_MAB_phenos_Left_Body_Length_csv['Fish Body Length']
Injected_MAB_phenos_Right_Body_Length= Injected_MAB_phenos_Right_Body_Length_csv['Fish Body Length']
Combined_Injected_MAB_phenos_Body_Length= pd.concat([Injected_MAB_phenos_Left_Body_Length, 
                                               Injected_MAB_phenos_Right_Body_Length])

# #     Injected MAB phenos Eye Area
Injected_MAB_phenos_Left_eye_area= Injected_MAB_phenos_Left_eye_paramaters['area']
Injected_MAB_phenos_Right_eye_area= Injected_MAB_phenos_Right_eye_paramaters['area']
Combined_Injected_MAB_phenos_eye_area = pd.concat([Injected_MAB_phenos_Left_eye_area, 
                                               Injected_MAB_phenos_Right_eye_area])

# #     Injected MAB phenos Body Length
Injected_MAB_phenos_Left_Body_Length= Injected_MAB_phenos_Left_Body_Length_csv['Fish Body Length']
Injected_MAB_phenos_Right_Body_Length= Injected_MAB_phenos_Right_Body_Length_csv['Fish Body Length']
Combined_Injected_MAB_phenos_Body_Length= pd.concat([Injected_MAB_phenos_Left_Body_Length, 
                                               Injected_MAB_phenos_Right_Body_Length])

# #     Injected MAB phenos Eye Major Axis Length
Injected_MAB_phenos_Left_eye_major_axis_length= Injected_MAB_phenos_Left_eye_paramaters['major_axis_length']
Injected_MAB_phenos_Right_eye_major_axis_length= Injected_MAB_phenos_Right_eye_paramaters['major_axis_length']
Combined_Injected_MAB_phenos_eye_major_axis_length = pd.concat([Injected_MAB_phenos_Left_eye_major_axis_length, 
                                               Injected_MAB_phenos_Right_eye_major_axis_length])



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

# # Control MAB siblings-----

Control_MAB_siblings_body_eye_ratios_Left= ( Control_MAB_siblings_Left_Body_Length / Control_MAB_siblings_Left_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Control_MAB_siblings_body_eye_ratios_Left.columns=['Body Length/ Eye Diameter Ratio']

Control_MAB_siblings_body_eye_ratios_Right= ( Control_MAB_siblings_Right_Body_Length / Control_MAB_siblings_Right_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Control_MAB_siblings_body_eye_ratios_Right.columns=['Body Length/ Eye Diameter Ratio']

Control_MAB_siblings_mean_eye_lengths = np.average([Control_MAB_siblings_Left_eye_major_axis_length, Control_MAB_siblings_Right_eye_major_axis_length], axis=0)

Control_MAB_siblings_mean_body_lengths = np.average([Control_MAB_siblings_Left_Body_Length, Control_MAB_siblings_Left_Body_Length], axis=0)

Control_MAB_siblings_body_eye_ratios= ( Control_MAB_siblings_mean_body_lengths / Control_MAB_siblings_mean_eye_lengths  * (8/3) ) * magnification_difference_in_pixels

Control_MAB_siblings_eye_body_ratio= 1 / Control_MAB_siblings_body_eye_ratios


##--------------------
# # Control MAB phenos-----
Control_MAB_phenos_body_eye_ratios_Left= ( Control_MAB_phenos_Left_Body_Length / Control_MAB_phenos_Left_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Control_MAB_phenos_body_eye_ratios_Left.columns=['Body Length/ Eye Diameter Ratio']

Control_MAB_phenos_body_eye_ratios_Right= ( Control_MAB_phenos_Right_Body_Length / Control_MAB_phenos_Right_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Control_MAB_phenos_body_eye_ratios_Right.columns=['Body Length/ Eye Diameter Ratio']

Control_MAB_phenos_mean_eye_lengths = np.average([Control_MAB_phenos_Left_eye_major_axis_length, Control_MAB_phenos_Right_eye_major_axis_length], axis=0)

Control_MAB_phenos_mean_body_lengths = np.average([Control_MAB_phenos_Left_Body_Length, Control_MAB_phenos_Left_Body_Length], axis=0)

Control_MAB_phenos_body_eye_ratios= ( Control_MAB_phenos_mean_body_lengths / Control_MAB_phenos_mean_eye_lengths  * (8/3) ) * magnification_difference_in_pixels

Control_MAB_phenos_eye_body_ratio= 1 / Control_MAB_phenos_body_eye_ratios


##--------------------
# # Injected MAB siblings-----

Injected_MAB_siblings_body_eye_ratios_Left= ( Injected_MAB_siblings_Left_Body_Length / Injected_MAB_siblings_Left_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Injected_MAB_siblings_body_eye_ratios_Left.columns=['Body Length/ Eye Diameter Ratio']

Injected_MAB_siblings_body_eye_ratios_Right= ( Injected_MAB_siblings_Right_Body_Length / Injected_MAB_siblings_Right_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Injected_MAB_siblings_body_eye_ratios_Right.columns=['Body Length/ Eye Diameter Ratio']

Injected_MAB_siblings_mean_eye_lengths = np.average([Injected_MAB_siblings_Left_eye_major_axis_length, Injected_MAB_siblings_Right_eye_major_axis_length], axis=0)

Injected_MAB_siblings_mean_body_lengths = np.average([Injected_MAB_siblings_Left_Body_Length, Injected_MAB_siblings_Left_Body_Length], axis=0)

Injected_MAB_siblings_body_eye_ratios= ( Injected_MAB_siblings_mean_body_lengths / Injected_MAB_siblings_mean_eye_lengths  * (8/3) ) * magnification_difference_in_pixels

Injected_MAB_siblings_eye_body_ratio= 1 / Injected_MAB_siblings_body_eye_ratios


##--------------------
# # Injected MAB phenos-----
Injected_MAB_phenos_body_eye_ratios_Left= ( Injected_MAB_phenos_Left_Body_Length / Injected_MAB_phenos_Left_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Injected_MAB_phenos_body_eye_ratios_Left.columns=['Body Length/ Eye Diameter Ratio']

Injected_MAB_phenos_body_eye_ratios_Right= ( Injected_MAB_phenos_Right_Body_Length / Injected_MAB_phenos_Right_eye_major_axis_length  * (8/3) ) * magnification_difference_in_pixels
Injected_MAB_phenos_body_eye_ratios_Right.columns=['Body Length/ Eye Diameter Ratio']

Injected_MAB_phenos_mean_eye_lengths = np.average([Injected_MAB_phenos_Left_eye_major_axis_length, Injected_MAB_phenos_Right_eye_major_axis_length], axis=0)

Injected_MAB_phenos_mean_body_lengths = np.average([Injected_MAB_phenos_Left_Body_Length, Injected_MAB_phenos_Left_Body_Length], axis=0)

Injected_MAB_phenos_body_eye_ratios= ( Injected_MAB_phenos_mean_body_lengths / Injected_MAB_phenos_mean_eye_lengths  * (8/3) ) * magnification_difference_in_pixels

Injected_MAB_phenos_eye_body_ratio= 1 / Injected_MAB_phenos_body_eye_ratios


##--------------------


#%% # Do Stats for Average Eye Length
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Average_Eye_Length.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Average Eye Lengths: Control and Injected
s1= Control_MAB_siblings_mean_eye_lengths   * Eye_conversion_8x
s2= Control_MAB_phenos_mean_eye_lengths     * Eye_conversion_8x
s3= Injected_MAB_siblings_mean_eye_lengths  * Eye_conversion_8x
s4= Injected_MAB_phenos_mean_eye_lengths    * Eye_conversion_8x
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Average Eye Length: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)

reportFile.write('Mean Control MAB siblings = ' + str(s1_mean) + '\n')
reportFile.write('Mean Control MAB phenos= ' + str(s2_mean) + '\n')
reportFile.write('Mean Injected MAB siblings = ' + str(s3_mean) + '\n')
reportFile.write('Mean Injected phenos siblings = '  + str(s4_mean) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

## --------------
## Mean Control MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s3)
reportFile.write( 'Means: Control MAB siblings vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s3)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB pheno vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s2, s4)
reportFile.write( 'Means: Control MAB pheno vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')


# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s2, s4)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB siblings vs Mean Control MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2)
reportFile.write( 'Means: Control MAB siblings vs Control MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n' )

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Injected MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s3, s4)
reportFile.write( 'Means: Injected MAB siblings vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s3, s4)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n' )

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB siblings vs Mean Injected MAB siblings
percentage_change= ((np.mean(s1) - np.mean(s3)) / np.mean(s1) ) *100 # in um
reportFile.write( 'Percentage Difference: ' + str(percentage_change) +  '(Mean Control MAB siblings ->  Mean Injected MAB sibllings)' + '\n' )

## Mean Control MAB pheno vs Mean Injected MAB phenos
percentage_change= ((np.mean(s2) - np.mean(s4)) / np.mean(s2) ) *100 # in um
reportFile.write( 'Percentage Difference: ' + str(percentage_change) +  '(Mean Control MAB phneos ->  Mean Injected MAB phenos)' + '\n' )


#-----------------
#------------------------
reportFile.close()



#%% # Plot Average Eye Length
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s1= Control_MAB_siblings_mean_eye_lengths   * Eye_conversion_8x
s2= Control_MAB_phenos_mean_eye_lengths     * Eye_conversion_8x
s3= Injected_MAB_siblings_mean_eye_lengths  * Eye_conversion_8x
s4= Injected_MAB_phenos_mean_eye_lengths    * Eye_conversion_8x

df1= pd.DataFrame({
    'Condition': ['Control MAB sibling', 'Control MAB phenos', 'Injected MAB sibling', 'Injected MAB phenos'], 
    'Lengths': [ s1, 
                 s2, 
                 s3, 
                 s4    ] })

df1=df1.explode('Lengths')

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(x= df1['Condition'], y= df1['Lengths'], data=df1, estimator='mean', errorbar=('se', 2), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Eye Length (µm)', fontsize=10 )
plt.title(' Average Eye Length: Control vs Injected')
# plt.ylim(260, 324)
plt.ylim(160, 340)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Length', dpi=900)  

plt.show()

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(x= df1['Condition'], y= df1['Lengths'], data=df1, estimator='mean', errorbar=('se', 2), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Body Length (µm)', fontsize=10 )
plt.title(' Average Body Length: Control vs Injected')
#plt.ylim(160, 340)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Length_Default', dpi=900)  

plt.show()


#------------------
# # Plot Boxplot with Scatter plot

s1= Control_MAB_siblings_mean_eye_lengths   * Eye_conversion_8x
s1a= pd.DataFrame([s1]).T
s1a.columns=['Control MAB siblings']
s1a = s1a.set_index(np.arange(0,len(s1a)))


s2= Control_MAB_phenos_mean_eye_lengths     * Eye_conversion_8x
s2a= pd.DataFrame([s2]).T
s2a.columns=['Control MAB phenos']
s2a = s2a.set_index(np.arange(0,len(s2a)))


s3= Injected_MAB_siblings_mean_eye_lengths  * Eye_conversion_8x
s3a= pd.DataFrame([s3]).T
s3a.columns=['Injected MAB siblings']
s3a = s3a.set_index(np.arange(0,len(s3a)))


s4= Injected_MAB_phenos_mean_eye_lengths    * Eye_conversion_8x
s4a= pd.DataFrame([s4]).T
s4a.columns=['Injected MAB phenos']
s4a = s4a.set_index(np.arange(0,len(s4a)))



df1= pd.DataFrame({
    'Condition': ['Control MAB sibling', 'Control MAB phenos', 'Injected MAB sibling', 'Injected MAB phenos'], 
    'Lengths': [ s1, 
                 s2, 
                 s3, 
                 s4    ] })


plt.figure(1)
  
ax= plt.subplot()

ax.boxplot(df1['Lengths'], labels= df1['Condition'],   showfliers=False,   widths=0.25, positions=range(len(df1['Condition'])))

df1=df1.explode('Lengths')
ax.scatter(y=df1['Lengths'], x=df1['Condition'], c= 'lightblue' , alpha=1.0)

plt.title('Average Eye Lengths')
plt.tight_layout()

ax.spines[['right', 'top']].set_visible(False)

    
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Lengths-Boxplot_with_scatter', dpi=900)  

plt.show()    


#%% # Do Stats for Average Body Length
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Average_Body_Lengths.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Average Body Lengths: Control and Injected
s1= Control_MAB_siblings_mean_body_lengths * Body_conversion_3x
s2= Control_MAB_phenos_mean_body_lengths * Body_conversion_3x
s3= Injected_MAB_siblings_mean_body_lengths * Body_conversion_3x
s4= Injected_MAB_phenos_mean_body_lengths * Body_conversion_3x
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Average Body Length: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)

reportFile.write('Mean Control MAB siblings = ' + str(s1_mean) + '\n')
reportFile.write('Mean Control MAB phenos= ' + str(s2_mean) + '\n')
reportFile.write('Mean Injected MAB siblings = ' + str(s3_mean) + '\n')
reportFile.write('Mean Injected phenos siblings = '  + str(s4_mean) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

## --------------
## Mean Control MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s3)
reportFile.write( 'Means: Control MAB siblings vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s3)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB pheno vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s2, s4)
reportFile.write( 'Means: Control MAB pheno vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')


# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s2, s4)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB siblings vs Mean Control MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2)
reportFile.write( 'Means: Control MAB siblings vs Control MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n' )

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Injected MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s3, s4)
reportFile.write( 'Means: Injected MAB siblings vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s3, s4)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n' )

reportFile.write('#------------------------' + '\n')

#-----------------
#-----------------
## Mean Control MAB siblings vs Mean Injected MAB siblings
percentage_change= ((s1_mean - s3_mean) / s1_mean ) *100 # in um
reportFile.write( 'Percentage Difference: ' + str(percentage_change) +  '(Mean Control MAB siblings ->  Mean Injected MAB sibllings)' + '\n' )

## Mean Control MAB pheno vs Mean Injected MAB phenos
percentage_change= ((s2_mean - s4_mean) / s2_mean ) *100 # in um
reportFile.write( 'Percentage Difference: ' + str(percentage_change) +  '(Mean Control MAB phenos ->  Mean Injected MAB phenos)' + '\n' + '\n' )



#-----------------
#------------------------
reportFile.close()


#%% # Plot Average Body Length
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s1= Control_MAB_siblings_mean_body_lengths * Body_conversion_3x
s2= Control_MAB_phenos_mean_body_lengths * Body_conversion_3x
s3= Injected_MAB_siblings_mean_body_lengths * Body_conversion_3x
s4= Injected_MAB_phenos_mean_body_lengths * Body_conversion_3x

df1= pd.DataFrame({
    'Condition': ['Control MAB sibling', 'Control MAB phenos', 'Injected MAB sibling', 'Injected MAB phenos'], 
    'Lengths': [ s1, 
                 s2, 
                 s3, 
                 s4    ] })

df1=df1.explode('Lengths')

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(x= df1['Condition'], y= df1['Lengths'], data=df1, estimator='mean', errorbar=('se', 2), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Body Length (µm)', fontsize=10 )
plt.title(' Average Body Length: Control vs Injected')
plt.ylim(1000, 4000)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Body_Length_Control_vs_Injected', dpi=900)  

plt.show()

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(x= df1['Condition'], y= df1['Lengths'], data=df1, estimator='mean', errorbar=('se', 2), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Body Length (µm)', fontsize=10 )
plt.title(' Average Body Length: Control vs Injected')
#plt.ylim(1000, 4000)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Body_Length_Control_vs_Injected_Default', dpi=900)  

plt.show()


#------------------
# # Plot Boxplot with Scatter plot

s1= Control_MAB_siblings_mean_body_lengths * Body_conversion_3x
s1a= pd.DataFrame([s1]).T
s1a.columns=['Control MAB siblings']
s1a = s1a.set_index(np.arange(0,len(s1a)))


s2= Control_MAB_phenos_mean_body_lengths  * Body_conversion_3x
s2a= pd.DataFrame([s2]).T
s2a.columns=['Control MAB phenos']
s2a = s2a.set_index(np.arange(0,len(s2a)))


s3= Injected_MAB_siblings_mean_body_lengths  * Body_conversion_3x
s3a= pd.DataFrame([s3]).T
s3a.columns=['Injected MAB siblings']
s3a = s3a.set_index(np.arange(0,len(s3a)))


s4= Injected_MAB_phenos_mean_body_lengths  * Body_conversion_3x
s4a= pd.DataFrame([s4]).T
s4a.columns=['Injected MAB phenos']
s4a = s4a.set_index(np.arange(0,len(s4a)))



df1= pd.DataFrame({
    'Condition': ['Control MAB sibling', 'Control MAB phenos', 'Injected MAB sibling', 'Injected MAB phenos'], 
    'Lengths': [ s1, 
                 s2, 
                 s3, 
                 s4    ] })


plt.figure(1)
  
ax= plt.subplot()

ax.boxplot(df1['Lengths'], labels= df1['Condition'],   showfliers=False,   widths=0.25, positions=range(len(df1['Condition'])))

df1=df1.explode('Lengths')
ax.scatter(y=df1['Lengths'], x=df1['Condition'], c= 'lightblue' , alpha=1.0)

plt.title('Average Body Lengths')
plt.tight_layout()

ax.spines[['right', 'top']].set_visible(False)

    
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Body_Lengths-Boxplot_with_scatter', dpi=900)  

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

# Control MAB siblings
Control_MAB_siblings_avg_eye_area= np.average([Combined_Control_MAB_siblings_eye_area], axis=0)
Control_MAB_siblings_avg_eye_area= Control_MAB_siblings_avg_eye_area * squared_single_pixel_width_in_um 

# Control MAB phenos
Control_MAB_phenos_avg_eye_area= np.average([Combined_Control_MAB_phenos_eye_area], axis=0)
Control_MAB_phenos_avg_eye_area= Control_MAB_phenos_avg_eye_area * squared_single_pixel_width_in_um  

# Injected MAB siblings
Injected_MAB_siblings_avg_eye_area= np.average([Combined_Injected_MAB_siblings_eye_area], axis=0)
Injected_MAB_siblings_avg_eye_area= Injected_MAB_siblings_avg_eye_area * squared_single_pixel_width_in_um 

# Injected MAB phenos
Injected_MAB_phenos_avg_eye_area= np.average([Combined_Injected_MAB_phenos_eye_area], axis=0)
Injected_MAB_phenos_avg_eye_area= Injected_MAB_phenos_avg_eye_area * squared_single_pixel_width_in_um  



#%% # Do Stats for Average Eye Area
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Average_Eye_Area.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Average Body Lengths: Control and Injected
s1= Control_MAB_siblings_avg_eye_area
s2= Control_MAB_phenos_avg_eye_area
s3= Injected_MAB_siblings_avg_eye_area
s4= Injected_MAB_phenos_avg_eye_area
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Average Eye Area: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)

reportFile.write('Mean Control MAB siblings = ' + str(s1_mean) + '\n')
reportFile.write('Mean Control MAB phenos= ' + str(s2_mean) + '\n')
reportFile.write('Mean Injected MAB siblings = ' + str(s3_mean) + '\n')
reportFile.write('Mean Injected phenos siblings = '  + str(s4_mean) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

## --------------
## Mean Control MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s3)
reportFile.write( 'Means: Control MAB siblings vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s3)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB pheno vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s2, s4)
reportFile.write( 'Means: Control MAB pheno vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')


# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s2, s4)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB siblings vs Mean Control MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2)
reportFile.write( 'Means: Control MAB siblings vs Control MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n' )

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Injected MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s3, s4)
reportFile.write( 'Means: Injected MAB siblings vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s3, s4)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n' )

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB siblings vs Mean Injected MAB siblings
percentage_change= ((np.mean(s1) - np.mean(s3)) / np.mean(s1) ) *100 # in um
reportFile.write( 'Percentage Difference: ' + str(percentage_change) +  '(Mean Control MAB siblings ->  Mean Injected MAB sibllings)' + '\n' )

## Mean Control MAB pheno vs Mean Injected MAB phenos
percentage_change= ((np.mean(s2) - np.mean(s4)) / np.mean(s2) ) *100 # in um
reportFile.write( 'Percentage Difference: ' + str(percentage_change) +  '(Mean Control MAB phneos ->  Mean Injected MAB phenos)' + '\n' )


#-----------------
#------------------------
reportFile.close()

#%% # Plot Average Eye Area
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s1= Control_MAB_siblings_avg_eye_area
s2= Control_MAB_phenos_avg_eye_area
s3= Injected_MAB_siblings_avg_eye_area
s4= Injected_MAB_phenos_avg_eye_area

df1= pd.DataFrame({
    'Condition': ['Control MAB sibling', 'Control MAB phenos', 'Injected MAB sibling', 'Injected MAB phenos'], 
    'Area': [ s1, 
                 s2, 
                 s3, 
                 s4    ] })

df1=df1.explode('Area')

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(x= df1['Condition'], y= df1['Area'], data=df1, estimator='mean', errorbar=('se', 2), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Eye Area (µm2)', fontsize=10 )
plt.title(' Average Eye Area: Control vs Injected')
plt.ylim(30000, 75000)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Area', dpi=900)  

plt.show()

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(x= df1['Condition'], y= df1['Area'], data=df1, estimator='mean', errorbar=('se', 2), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Eye Area (µm2)', fontsize=10 )
plt.title(' Average Eye Area: Control vs Injected')
#plt.ylim(30000, 75000)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Area_Default', dpi=900)  

plt.show()


#------------------
# # Plot Boxplot with Scatter plot

s1= Control_MAB_siblings_avg_eye_area
s1a= pd.DataFrame([s1]).T
s1a.columns=['Control MAB siblings']
s1a = s1a.set_index(np.arange(0,len(s1a)))


s2= Control_MAB_phenos_avg_eye_area
s2a= pd.DataFrame([s2]).T
s2a.columns=['Control MAB phenos']
s2a = s2a.set_index(np.arange(0,len(s2a)))


s3= Injected_MAB_siblings_avg_eye_area
s3a= pd.DataFrame([s3]).T
s3a.columns=['Injected MAB siblings']
s3a = s3a.set_index(np.arange(0,len(s3a)))


s4= Injected_MAB_phenos_avg_eye_area
s4a= pd.DataFrame([s4]).T
s4a.columns=['Injected MAB phenos']
s4a = s4a.set_index(np.arange(0,len(s4a)))



df1= pd.DataFrame({
    'Condition': ['Control MAB sibling', 'Control MAB phenos', 'Injected MAB sibling', 'Injected MAB phenos'], 
    'Area': [ s1, 
                 s2, 
                 s3, 
                 s4    ] })


plt.figure(1)
  
ax= plt.subplot()

ax.boxplot(df1['Area'], labels= df1['Condition'],   showfliers=False,   widths=0.25, positions=range(len(df1['Condition'])))

df1=df1.explode('Area')
ax.scatter(y=df1['Area'], x=df1['Condition'], c= 'lightblue' , alpha=1.0)

plt.title('Average Eye Area')
plt.tight_layout()

ax.spines[['right', 'top']].set_visible(False)

    
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Area-Boxplot_with_scatter', dpi=900)  

plt.show()    




#%% # Plot Eccentricity
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s1= Combined_Control_MAB_siblings_eye_eccentricity
s2= Combined_Control_MAB_phenos_eye_eccentricity
s3= Combined_Injected_MAB_siblings_eye_eccentricity
s4= Combined_Injected_MAB_phenos_eye_eccentricity

df1= pd.DataFrame({
    'Condition': ['Control MAB sibling', 'Control MAB phenos', 'Injected MAB sibling', 'Injected MAB phenos'], 
    'Area': [ s1, 
                 s2, 
                 s3, 
                 s4    ] })

df1=df1.explode('Area')

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(x= df1['Condition'], y= df1['Area'], data=df1, estimator='mean', errorbar=('se', 2), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Eccentricity', fontsize=10 )
plt.title(' Eccentricity: Control vs Injected')
#plt.ylim(30000, 75000)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Eccentricity', dpi=900)  

plt.show()

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(x= df1['Condition'], y= df1['Area'], data=df1, estimator='mean', errorbar=('se', 2), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Eye Area (µm2)', fontsize=10 )
plt.title(' Average Eye Area: Control vs Injected')
#plt.ylim(30000, 75000)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Area_Default', dpi=900)  

plt.show()


#------------------
# # Plot Boxplot with Scatter plot

s1= Control_MAB_siblings_avg_eye_area
s1a= pd.DataFrame([s1]).T
s1a.columns=['Control MAB siblings']
s1a = s1a.set_index(np.arange(0,len(s1a)))


s2= Control_MAB_phenos_avg_eye_area
s2a= pd.DataFrame([s2]).T
s2a.columns=['Control MAB phenos']
s2a = s2a.set_index(np.arange(0,len(s2a)))


s3= Injected_MAB_siblings_avg_eye_area
s3a= pd.DataFrame([s3]).T
s3a.columns=['Injected MAB siblings']
s3a = s3a.set_index(np.arange(0,len(s3a)))


s4= Injected_MAB_phenos_avg_eye_area
s4a= pd.DataFrame([s4]).T
s4a.columns=['Injected MAB phenos']
s4a = s4a.set_index(np.arange(0,len(s4a)))



df1= pd.DataFrame({
    'Condition': ['Control MAB sibling', 'Control MAB phenos', 'Injected MAB sibling', 'Injected MAB phenos'], 
    'Area': [ s1, 
                 s2, 
                 s3, 
                 s4    ] })


plt.figure(1)
  
ax= plt.subplot()

ax.boxplot(df1['Area'], labels= df1['Condition'],   showfliers=False,   widths=0.25, positions=range(len(df1['Condition'])))

df1=df1.explode('Area')
ax.scatter(y=df1['Area'], x=df1['Condition'], c= 'lightblue' , alpha=1.0)

plt.title('Average Eye Area')
plt.tight_layout()

ax.spines[['right', 'top']].set_visible(False)

    
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Area-Boxplot_with_scatter', dpi=900)  

plt.show()    




#%% # Do Stats for Eccentricity
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Eccentricity.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Average Body Lengths: Control and Injected

s1= Combined_Control_MAB_siblings_eye_eccentricity
s2= Combined_Control_MAB_phenos_eye_eccentricity
s3= Combined_Injected_MAB_siblings_eye_eccentricity
s4= Combined_Injected_MAB_phenos_eye_eccentricity

  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Average Eye Area: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)

reportFile.write('Mean Control MAB siblings = ' + str(s1_mean) + '\n')
reportFile.write('Mean Control MAB phenos= ' + str(s2_mean) + '\n')
reportFile.write('Mean Injected MAB siblings = ' + str(s3_mean) + '\n')
reportFile.write('Mean Injected phenos siblings = '  + str(s4_mean) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

## --------------
## Mean Control MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s3)
reportFile.write( 'Means: Control MAB siblings vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s3)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB pheno vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s2, s4)
reportFile.write( 'Means: Control MAB pheno vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')


# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s2, s4)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB siblings vs Mean Control MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2)
reportFile.write( 'Means: Control MAB siblings vs Control MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n' )

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Injected MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s3, s4)
reportFile.write( 'Means: Injected MAB siblings vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s3, s4)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n' )

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB siblings vs Mean Injected MAB siblings
percentage_change= ((np.mean(s1) - np.mean(s3)) / np.mean(s1) ) *100 # in um
reportFile.write( 'Percentage Difference: ' + str(percentage_change) +  '(Mean Control MAB siblings ->  Mean Injected MAB sibllings)' + '\n' )

## Mean Control MAB pheno vs Mean Injected MAB phenos
percentage_change= ((np.mean(s2) - np.mean(s4)) / np.mean(s2) ) *100 # in um
reportFile.write( 'Percentage Difference: ' + str(percentage_change) +  '(Mean Control MAB phneos ->  Mean Injected MAB phenos)' + '\n' )


#-----------------
#------------------------
reportFile.close()







#%% # Do Stats for Average Eye Body Ratio
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# # Create report file
# reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
#                   r'/Stats_Report_Average_Eye_Body_Ratio_Control_vs_Injected.txt')
# reportFile = open(reportFilename, 'w')

# #-----------
# # Average Eye Body Ratio: Control and Injected
# s7= Control_eye_body_ratio
# s8= Injected_eye_body_ratio
  
# reportFile.write( '\n' + '#---------------------------' + '\n' )
# reportFile.write('Average Eye Body Ratio: Control vs Injected' + '\n' + '\n')  

# # Find means
# s7_mean = np.mean(s7)
# s8_mean = np.mean(s8)
# reportFile.write('Mean Uninjected = ' + str(s7_mean) + '\n')
# reportFile.write('Mean Injected = '  + str(s8_mean) + '\n')

# # Unpaired t-test
# t, p = sp.stats.ttest_ind(s7, s8)
# reportFile.write('\n' + 'Unpaired t-test' +'\n' +
#                  't= ' + str(t) +'\n' +
#                  'p= ' + str(p) + '\n')

# # NonParametric Mannwhitney U-test
# u, p = sp.stats.mannwhitneyu(s7, s8)
# reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
#                  'u= ' + str(u) +'\n' +
#                  'p= ' + str(p) + '\n')

# #------------------------
# reportFile.close()


# #%% # Plot Average Eye Body Ratio 
# #--------------------------------------------------------------------------
# #--------------------------------------------------------------------------

# s7= Control_eye_body_ratio
# s7= pd.DataFrame([s7])

# s8= Injected_eye_body_ratio
# s8= pd.DataFrame([s8])

# df4= pd.concat([ s7, s8 ], axis=0).T
# df4.columns=['Control', 'Injected']

# #-------------------------
# # # Plot Figure
# plt.Figure()

# ax=sns.barplot(data=df4, estimator='mean', errorbar=('se', 2), capsize=.2, errwidth=2.1, color='lightblue', width=.45) 
# ax.spines[['right', 'top']].set_visible(False)

# plt.xlabel('Condition', fontsize=10)
# plt.ylabel('Eye/ Body Ratio', fontsize=10 )
# plt.title(' Average Eye Body Ratio: Control vs Injected')
# plt.ylim(0.045, 0.105)

# plt.tight_layout()

# # Save figure
# plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Body_Ratio_Control_vs_Injected', dpi=900)  

# plt.show()

# #-------------------------
# # # Plot Figure
# plt.Figure()

# ax=sns.barplot(data=df4, estimator='mean', errorbar=('se', 2), capsize=.2, errwidth=2.1, color='lightblue', width=.45) 
# ax.spines[['right', 'top']].set_visible(False)

# plt.xlabel('Condition', fontsize=10)
# plt.ylabel('Eye/ Body Ratio', fontsize=10 )
# plt.title(' Average Eye Body Ratio: Control vs Injected')
# # plt.ylim(0.045, 0.105)

# plt.tight_layout()

# # Save figure
# plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Body_Ratio_Control_vs_Injected_Default', dpi=900)  

# plt.show()

# #%% # Not in use --> Double checking the eye/body ratio to be extra sure 
# #--------------------------------------------------------------------------
# #--------------------------------------------------------------------------

# # x= Control_mean_body_lengths * Body_conversion_3x 
# # y= Control_mean_eye_lengths * Eye_conversion_8x
# # control_ratio= y/x
# # control_ratios= np.average( pd.DataFrame([control_ratio]), axis=1)

# # x= Injected_mean_body_lengths * Body_conversion_3x 
# # y= Injected_mean_eye_lengths * Eye_conversion_8x
# # injected_ratio= y/x
# # injected_ratios= np.average( pd.DataFrame([injected_ratio]), axis=1)


# # s9= control_ratio
# # s10= injected_ratio

# # t, p1 = sp.stats.ttest_ind(s9, s10)
# # u, p2 = sp.stats.mannwhitneyu(s9, s10)


# #%% # List of used websites

# '''
# Used websites:
#     https://www.otago.ac.nz/__data/assets/pdf_file/0028/301789/spatial-calibration-of-an-image-684707.pdf
#     https://monashmicroimaging.github.io/gitbook-fiji-basics/part_4_scales_and_sizing.html
#     https://forum.image.sc/t/regionprops-area-conversion-to-micron-unit-of-measure/84779/2
#     https://imagej.net/ij/docs/menus/analyze.html#:~:text=Area%20%2D%20Area%20of%20selection%20in,gray%20value%20within%20the%20selection.
#     https://scikit-image.org/docs/stable/api/skimage.measure.html
# '''


# #%% Finish


