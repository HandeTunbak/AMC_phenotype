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

# #    Control MAB sibling Head Length
Control_MAB_siblings_Left_Head_Length= Control_MAB_siblings_Left_Body_Length_csv['Fish Head Length']
Control_MAB_siblings_Right_Head_Length= Control_MAB_siblings_Right_Body_Length_csv['Fish Head Length']
Combined_Control_MAB_siblings_Head_Length= pd.concat([Control_MAB_siblings_Left_Head_Length, 
                                               Control_MAB_siblings_Right_Head_Length])

# #    Control MAB sibling Tail Length
Control_MAB_siblings_Left_Tail_Length= Control_MAB_siblings_Left_Body_Length_csv['Fish Tail Length']
Control_MAB_siblings_Right_Tail_Length= Control_MAB_siblings_Right_Body_Length_csv['Fish Tail Length']
Combined_Control_MAB_siblings_Tail_Length= pd.concat([Control_MAB_siblings_Left_Tail_Length, 
                                               Control_MAB_siblings_Right_Tail_Length])

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

# #     Control MAB phenos Head Length
Control_MAB_phenos_Left_Head_Length= Control_MAB_phenos_Left_Body_Length_csv['Fish Head Length']
Control_MAB_phenos_Right_Head_Length= Control_MAB_phenos_Right_Body_Length_csv['Fish Head Length']
Combined_Control_MAB_phenos_Head_Length= pd.concat([Control_MAB_phenos_Left_Head_Length, 
                                               Control_MAB_phenos_Right_Head_Length])

# #     Control MAB phenos Tail Length
Control_MAB_phenos_Left_Tail_Length= Control_MAB_phenos_Left_Body_Length_csv['Fish Tail Length']
Control_MAB_phenos_Right_Tail_Length= Control_MAB_phenos_Right_Body_Length_csv['Fish Tail Length']
Combined_Control_MAB_phenos_Tail_Length= pd.concat([Control_MAB_phenos_Left_Tail_Length, 
                                               Control_MAB_phenos_Right_Tail_Length])

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

# #    Injected MAB sibling Head Length
Injected_MAB_siblings_Left_Head_Length= Injected_MAB_siblings_Left_Body_Length_csv['Fish Head Length']
Injected_MAB_siblings_Right_Head_Length= Injected_MAB_siblings_Right_Body_Length_csv['Fish Head Length']
Combined_Injected_MAB_siblings_Head_Length= pd.concat([Injected_MAB_siblings_Left_Head_Length, 
                                               Injected_MAB_siblings_Right_Head_Length])

# #    Injected MAB sibling Tail Length
Injected_MAB_siblings_Left_Tail_Length= Injected_MAB_siblings_Left_Body_Length_csv['Fish Tail Length']
Injected_MAB_siblings_Right_Tail_Length= Injected_MAB_siblings_Right_Body_Length_csv['Fish Tail Length']
Combined_Injected_MAB_siblings_Tail_Length= pd.concat([Injected_MAB_siblings_Left_Tail_Length, 
                                               Injected_MAB_siblings_Right_Tail_Length])

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

# #     Injected MAB phenos Head Length
Injected_MAB_phenos_Left_Head_Length= Injected_MAB_phenos_Left_Body_Length_csv['Fish Head Length']
Injected_MAB_phenos_Right_Head_Length= Injected_MAB_phenos_Right_Body_Length_csv['Fish Head Length']
Combined_Injected_MAB_phenos_Head_Length= pd.concat([Injected_MAB_phenos_Left_Head_Length, 
                                               Injected_MAB_phenos_Right_Head_Length])

# #     Injected MAB phenos Tail Length
Injected_MAB_phenos_Left_Tail_Length= Injected_MAB_phenos_Left_Body_Length_csv['Fish Tail Length']
Injected_MAB_phenos_Right_Tail_Length= Injected_MAB_phenos_Right_Body_Length_csv['Fish Tail Length']
Combined_Injected_MAB_phenos_Tail_Length= pd.concat([Injected_MAB_phenos_Left_Tail_Length, 
                                               Injected_MAB_phenos_Right_Tail_Length])

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


#%% # Do Stats for Eye Length
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_All_Eye_Length_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Eye Lengths: Control and Injected
s1= Combined_Control_MAB_siblings_eye_major_axis_length * Eye_conversion_8x
s2= Combined_Control_MAB_phenos_eye_major_axis_length * Eye_conversion_8x
s3= Combined_Injected_MAB_siblings_eye_major_axis_length * Eye_conversion_8x
s4= Combined_Injected_MAB_phenos_eye_major_axis_length * Eye_conversion_8x
  
reportFile.write('All Eye Length: Control vs Injected' + '\n' )  
reportFile.write( '#---------------------------' + '\n' )
reportFile.write( '#---------------------------' + '\n' + '\n')

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)

reportFile.write('Mean Control MAB siblings = '      + str(s1_mean) + ' um' +'\n')
reportFile.write('Mean Control MAB phenos= '         + str(s2_mean) +' um' + '\n')
reportFile.write('Mean Injected MAB siblings = '     + str(s3_mean) +' um' + '\n')
reportFile.write('Mean Injected MAB phenos = '       + str(s4_mean) + ' um' +'\n' + '\n')

reportFile.write('#------------------------' + '\n')

## --------------
## Mean Control MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s3, equal_var = False)
reportFile.write( 'Means: Control MAB siblings vs Injected MAB siblings' + 
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
t, p = sp.stats.ttest_ind(s2, s4, equal_var = False)
reportFile.write( 'Means: Control MAB phenos vs Injected MAB phenos' + 
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
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
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
t, p = sp.stats.ttest_ind(s3, s4, equal_var = False)
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
reportFile.write( 'Mean Control MAB siblings ->  Mean Injected MAB sibllings' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %'+  '\n' + '\n' )

## Mean Control MAB pheno vs Mean Injected MAB phenos
percentage_change= ((np.mean(s2) - np.mean(s4)) / np.mean(s2) ) *100 # in um
reportFile.write( 'Mean Control MAB phenos ->  Mean Injected MAB phenos' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %' + '\n' )


#-----------------
#------------------------
reportFile.close()

#%% Save Eye Lengths as CSV

# Control
s1= Combined_Control_MAB_siblings_eye_major_axis_length * Eye_conversion_8x
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= Combined_Control_MAB_phenos_eye_major_axis_length * Eye_conversion_8x
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

# Injected
s3= Combined_Injected_MAB_siblings_eye_major_axis_length * Eye_conversion_8x
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))

s4= Combined_Injected_MAB_phenos_eye_major_axis_length * Eye_conversion_8x
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))



# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')

table = data_df1

table.to_csv('C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep' + 
             '/Eye_Lengths_(um).csv' , index=False)

#%% # Box Plot with Scatter all Eye Lengths 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Control
s1= Combined_Control_MAB_siblings_eye_major_axis_length * Eye_conversion_8x
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= Combined_Control_MAB_phenos_eye_major_axis_length * Eye_conversion_8x
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

# Injected
s3= Combined_Injected_MAB_siblings_eye_major_axis_length * Eye_conversion_8x
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))

s4= Combined_Injected_MAB_phenos_eye_major_axis_length * Eye_conversion_8x
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))



# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')

#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

vals[0]=vals[0][~np.isnan(vals[0])]
xs[0]= xs[0][0:len(vals[0])]

vals[1]=vals[1][~np.isnan(vals[1])]
xs[1]= xs[1][0:len(vals[1])]

vals[2]=vals[2][~np.isnan(vals[2])]
xs[2]= xs[2][0:len(vals[2])]

vals[3]=vals[3][~np.isnan(vals[3])]
xs[3]= xs[3][0:len(vals[3])]


plt.figure(1)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', 'black', 'black', ]
markers= ['.', '^', '.', '^']
sizes= [45, 20, 45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Eye Length')
    plt.tight_layout()
    plt.ylabel('Eye Length (um)')
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    
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

# Control
Combined_Control_MAB_siblings_eye_area= Combined_Control_MAB_siblings_eye_area * squared_single_pixel_width_in_um 
Combined_Control_MAB_phenos_eye_area=   Combined_Control_MAB_phenos_eye_area * squared_single_pixel_width_in_um 

# Injected
Combined_Injected_MAB_siblings_eye_area= Combined_Injected_MAB_siblings_eye_area * squared_single_pixel_width_in_um 
Combined_Injected_MAB_phenos_eye_area=   Combined_Injected_MAB_phenos_eye_area * squared_single_pixel_width_in_um 


#%% # Do Stats for Eye Areas
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_All_Eye_Area_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Eye Areas: Control and Injected
s1= Combined_Control_MAB_siblings_eye_area
s2= Combined_Control_MAB_phenos_eye_area
s3= Combined_Injected_MAB_siblings_eye_area
s4= Combined_Injected_MAB_phenos_eye_area
  
reportFile.write('All Eye Area: Control vs Injected' + '\n' )  
reportFile.write( '#---------------------------' + '\n' )
reportFile.write( '#---------------------------' + '\n' + '\n')

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)

reportFile.write('Mean Control MAB siblings = '      + str(s1_mean) + ' um^2' +'\n')
reportFile.write('Mean Control MAB phenos= '         + str(s2_mean) +' um^2' + '\n')
reportFile.write('Mean Injected MAB siblings = '     + str(s3_mean) +' um^2' + '\n')
reportFile.write('Mean Injected MAB phenos = '  + str(s4_mean) + ' um^2' +'\n' + '\n')

reportFile.write('#------------------------' + '\n')

## --------------
## Mean Control MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s3, equal_var = False)
reportFile.write( 'Means: Control MAB siblings vs Injected MAB siblings' + 
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
t, p = sp.stats.ttest_ind(s2, s4, equal_var = False)
reportFile.write( 'Means: Control MAB phenos vs Injected MAB phenos' + 
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
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
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
t, p = sp.stats.ttest_ind(s3, s4, equal_var = False)
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
percentage_change= ((np.mean(s1) - np.mean(s3)) / np.mean(s1) ) *100 # in um2
reportFile.write( 'Mean Control MAB siblings ->  Mean Injected MAB sibllings' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %'+  '\n' + '\n' )

## Mean Control MAB pheno vs Mean Injected MAB phenos
percentage_change= ((np.mean(s2) - np.mean(s4)) / np.mean(s2) ) *100 # in um2
reportFile.write( 'Mean Control MAB phenos ->  Mean Injected MAB phenos' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %' + '\n' )


#-----------------
#------------------------
reportFile.close()



#%% Save Eye Area as CSV

# Control
s1= Combined_Control_MAB_siblings_eye_area
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= Combined_Control_MAB_phenos_eye_area
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

# Injected
s3= Combined_Injected_MAB_siblings_eye_area
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))

s4= Combined_Injected_MAB_phenos_eye_area
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))



# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')

table = data_df1

table.to_csv('C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep' +
             '/Eye_Area_(um^2).csv' , index=False)



#%% # Box Plot with Scatter all Eye Area 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Control
s1= Combined_Control_MAB_siblings_eye_area
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= Combined_Control_MAB_phenos_eye_area
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

# Injected
s3= Combined_Injected_MAB_siblings_eye_area
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))

s4= Combined_Injected_MAB_phenos_eye_area
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))



# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')

#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

vals[0]=vals[0][~np.isnan(vals[0])]
xs[0]= xs[0][0:len(vals[0])]

vals[1]=vals[1][~np.isnan(vals[1])]
xs[1]= xs[1][0:len(vals[1])]

vals[2]=vals[2][~np.isnan(vals[2])]
xs[2]= xs[2][0:len(vals[2])]

vals[3]=vals[3][~np.isnan(vals[3])]
xs[3]= xs[3][0:len(vals[3])]


plt.figure(1)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', 'black', 'black', ]
markers= ['.', '^', '.', '^']
sizes= [45, 20, 45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Eye Area')
    plt.tight_layout()
    plt.ylabel('Eye Area (um^2)')
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/All_Eye_Areas-Boxplot_with_scatter', dpi=900)  

plt.show()  


#%% # Do Stats for Average Body Length (Average Body Length per Fish)
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Control_vs_Injected_Average_Body_Length.txt')
reportFile = open(reportFilename, 'w')


#-----------
# Average Body Lengths: Control and Injected
# Control
s1= pd.concat([Control_MAB_siblings_Left_Body_Length, Control_MAB_siblings_Right_Body_Length ], axis=1)
s1= s1 * Body_conversion_3x    
s1= np.mean(s1, axis=1)

s2= pd.concat([Control_MAB_phenos_Left_Body_Length, Control_MAB_phenos_Right_Body_Length ], axis=1)
s2= s2 * Body_conversion_3x    
s2= np.mean(s2, axis=1)

# Injected
s3= pd.concat([Injected_MAB_siblings_Left_Body_Length, Injected_MAB_siblings_Right_Body_Length ], axis=1)
s3= s3 * Body_conversion_3x    
s3= np.mean(s3, axis=1)

s4= pd.concat([Injected_MAB_phenos_Left_Body_Length, Injected_MAB_phenos_Right_Body_Length ], axis=1)
s4= s4 * Body_conversion_3x    
s4= np.mean(s4, axis=1)


reportFile.write('Average Body Length: Control vs Injected' + '\n' )  
reportFile.write( '#---------------------------' + '\n' )
reportFile.write( '#---------------------------' + '\n' + '\n')

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)

reportFile.write('Mean Control MAB siblings = '      + str(s1_mean) + ' um' +'\n')
reportFile.write('Mean Control MAB phenos= '         + str(s2_mean) +' um' + '\n')
reportFile.write('Mean Injected MAB siblings = '     + str(s3_mean) +' um' + '\n')
reportFile.write('Mean Injected MAB phenos = '       + str(s4_mean) + ' um' +'\n' + '\n')

reportFile.write('#------------------------' + '\n')

## --------------
## Mean Control MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s3, equal_var = False)
reportFile.write( 'Means: Control MAB siblings vs Injected MAB siblings' + 
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
## Mean Control MAB phenos vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s2, s4, equal_var = False)
reportFile.write( 'Means: Control MAB phenos vs Injected MAB phenos' + 
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
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
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
t, p = sp.stats.ttest_ind(s3, s4, equal_var = False)
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
reportFile.write( 'Mean Control MAB siblings ->  Mean Injected MAB sibllings' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %'+  '\n' + '\n' )

## Mean Control MAB pheno vs Mean Injected MAB phenos
percentage_change= ((np.mean(s2) - np.mean(s4)) / np.mean(s2) ) *100 # in um
reportFile.write( 'Mean Control MAB phenos ->  Mean Injected MAB phenos' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %' + '\n' )


#-----------------
#------------------------
reportFile.close()


#%% Save Body Lengths as CSV

# Control
s1= Combined_Control_MAB_siblings_Body_Length
s1= s1 * Body_conversion_3x    
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= Combined_Control_MAB_phenos_Body_Length
s2= s2 * Body_conversion_3x    
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

# Injected
s3= Combined_Injected_MAB_siblings_Body_Length
s3= s3 * Body_conversion_3x    
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))

s4= Combined_Injected_MAB_phenos_Body_Length
s4= s4 * Body_conversion_3x    
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))


# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')
table = data_df1

table.to_csv('C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep' + 
             '/Body_Length_(um).csv' , index=False)


#%% Save Average Body Lengths as CSV

# Control
s1= pd.concat([Control_MAB_siblings_Left_Body_Length, Control_MAB_siblings_Right_Body_Length ], axis=1)
s1= s1 * Body_conversion_3x    
s1= np.mean(s1, axis=1)
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= pd.concat([Control_MAB_phenos_Left_Body_Length, Control_MAB_phenos_Right_Body_Length ], axis=1)
s2= s2 * Body_conversion_3x    
s2= np.mean(s2, axis=1)
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

# Injected
s3= pd.concat([Injected_MAB_siblings_Left_Body_Length, Injected_MAB_siblings_Right_Body_Length ], axis=1)
s3= s3 * Body_conversion_3x    
s3= np.mean(s3, axis=1)
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))

s4= pd.concat([Injected_MAB_phenos_Left_Body_Length, Injected_MAB_phenos_Right_Body_Length ], axis=1)
s4= s4 * Body_conversion_3x    
s4= np.mean(s4, axis=1)
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))


# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')
table = data_df1

table.to_csv('C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep' + 
             '/Average_Body_Length_(um).csv' , index=False)


#%% # Box Plot with Scatter Average Body Length 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Control
s1= pd.concat([Control_MAB_siblings_Left_Body_Length, Control_MAB_siblings_Right_Body_Length ], axis=1)
s1= s1 * Body_conversion_3x    
s1= np.mean(s1, axis=1)
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= pd.concat([Control_MAB_phenos_Left_Body_Length, Control_MAB_phenos_Right_Body_Length ], axis=1)
s2= s2 * Body_conversion_3x    
s2= np.mean(s2, axis=1)
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

# Injected
s3= pd.concat([Injected_MAB_siblings_Left_Body_Length, Injected_MAB_siblings_Right_Body_Length ], axis=1)
s3= s3 * Body_conversion_3x    
s3= np.mean(s3, axis=1)
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))

s4= pd.concat([Injected_MAB_phenos_Left_Body_Length, Injected_MAB_phenos_Right_Body_Length ], axis=1)
s4= s4 * Body_conversion_3x    
s4= np.mean(s4, axis=1)
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))


# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')

#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

vals[0]=vals[0][~np.isnan(vals[0])]
xs[0]= xs[0][0:len(vals[0])]

vals[1]=vals[1][~np.isnan(vals[1])]
xs[1]= xs[1][0:len(vals[1])]

vals[2]=vals[2][~np.isnan(vals[2])]
xs[2]= xs[2][0:len(vals[2])]

vals[3]=vals[3][~np.isnan(vals[3])]
xs[3]= xs[3][0:len(vals[3])]


plt.figure(1)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', 'black', 'black', ]
markers= ['.', '^', '.', '^']
sizes= [45, 20, 45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Average Body Length')
    plt.tight_layout()
    plt.ylabel('Body Length (um)')
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    
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

# Control
s1= Combined_Control_MAB_siblings_eye_eccentricity
s2= Combined_Control_MAB_phenos_eye_eccentricity

# Injected
s3= Combined_Injected_MAB_siblings_eye_eccentricity
s4= Combined_Injected_MAB_phenos_eye_eccentricity
  
reportFile.write('All Eye Eccentricity: Control vs Injected' + '\n' )  
reportFile.write( '#---------------------------' + '\n' )
reportFile.write( '#---------------------------' + '\n' + '\n')

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)

reportFile.write('Mean Control MAB siblings = '      + str(s1_mean) + ' um' +'\n')
reportFile.write('Mean Control MAB phenos= '         + str(s2_mean) +' um' + '\n')
reportFile.write('Mean Injected MAB siblings = '     + str(s3_mean) +' um' + '\n')
reportFile.write('Mean Injected MAB phenos = '       + str(s4_mean) + ' um' +'\n' + '\n')

reportFile.write('#------------------------' + '\n')

## --------------
## Mean Control MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s3, equal_var = False)
reportFile.write( 'Means: Control MAB siblings vs Injected MAB siblings' + 
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
t, p = sp.stats.ttest_ind(s2, s4, equal_var = False)
reportFile.write( 'Means: Control MAB phenos vs Injected MAB phenos' + 
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
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
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
t, p = sp.stats.ttest_ind(s3, s4, equal_var = False)
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
percentage_change= ((np.mean(s1) - np.mean(s3)) / np.mean(s1) ) *100
reportFile.write( 'Mean Control MAB siblings ->  Mean Injected MAB sibllings' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %'+  '\n' + '\n' )

## Mean Control MAB pheno vs Mean Injected MAB phenos
percentage_change= ((np.mean(s2) - np.mean(s4)) / np.mean(s2) ) *100
reportFile.write( 'Mean Control MAB phenos ->  Mean Injected MAB phenos' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %' + '\n' )


#-----------------
#------------------------
reportFile.close()


#%% Save Eccentricty as CSV

# Control
s1= Combined_Control_MAB_siblings_eye_eccentricity
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= Combined_Control_MAB_phenos_eye_eccentricity
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

# Injected
s3= Combined_Injected_MAB_siblings_eye_eccentricity
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))

s4= Combined_Injected_MAB_phenos_eye_eccentricity
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))


# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')


table = data_df1

table.to_csv('C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep' + 
             '/Eccentricity.csv' , index=False)


#%% # Box Plot with Scatter all Eye Eccentricity 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Control
s1= Combined_Control_MAB_siblings_eye_eccentricity
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= Combined_Control_MAB_phenos_eye_eccentricity
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

# Injected
s3= Combined_Injected_MAB_siblings_eye_eccentricity
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))

s4= Combined_Injected_MAB_phenos_eye_eccentricity
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))


# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')

#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

vals[0]=vals[0][~np.isnan(vals[0])]
xs[0]= xs[0][0:len(vals[0])]

vals[1]=vals[1][~np.isnan(vals[1])]
xs[1]= xs[1][0:len(vals[1])]

vals[2]=vals[2][~np.isnan(vals[2])]
xs[2]= xs[2][0:len(vals[2])]

vals[3]=vals[3][~np.isnan(vals[3])]
xs[3]= xs[3][0:len(vals[3])]


plt.figure(1)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', 'black', 'black', ]
markers= ['.', '^', '.', '^']
sizes= [45, 20, 45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Eye Eccentricity')
    plt.tight_layout()
    plt.ylabel('Eccentricty')
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/All_Eye_Eccentricty-Boxplot_with_scatter', dpi=900)  

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

# Control
s1a= Combined_Control_MAB_siblings_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s1b= Combined_Control_MAB_siblings_Body_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once

s2a= Combined_Control_MAB_phenos_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s2b= Combined_Control_MAB_phenos_Body_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once


# Injected
s3a= Combined_Injected_MAB_siblings_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s3b= Combined_Injected_MAB_siblings_Body_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once

s4a= Combined_Injected_MAB_phenos_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s4b= Combined_Injected_MAB_phenos_Body_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once


s1= s1a/s1b
s2= s2a/s2b
s3= s3a/s3b
s4= s4a/s4b


reportFile.write('Eye Body Ratio: Control vs Injected' + '\n' )  
reportFile.write( '#---------------------------' + '\n' )
reportFile.write( '#---------------------------' + '\n' + '\n')

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)

reportFile.write('Mean Control MAB siblings = '      + str(s1_mean) + ' um' +'\n')
reportFile.write('Mean Control MAB phenos= '         + str(s2_mean) +' um' + '\n')
reportFile.write('Mean Injected MAB siblings = '     + str(s3_mean) +' um' + '\n')
reportFile.write('Mean Injected MAB phenos = '       + str(s4_mean) + ' um' +'\n' + '\n')

reportFile.write('#------------------------' + '\n')

## --------------
## Mean Control MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s3, equal_var = False)
reportFile.write( 'Means: Control MAB siblings vs Injected MAB siblings' + 
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
t, p = sp.stats.ttest_ind(s2, s4, equal_var = False)
reportFile.write( 'Means: Control MAB phenos vs Injected MAB phenos' + 
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
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
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
t, p = sp.stats.ttest_ind(s3, s4, equal_var = False)
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
percentage_change= ((np.mean(s1) - np.mean(s3)) / np.mean(s1) ) *100
reportFile.write( 'Mean Control MAB siblings ->  Mean Injected MAB sibllings' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %'+  '\n' + '\n' )

## Mean Control MAB pheno vs Mean Injected MAB phenos
percentage_change= ((np.mean(s2) - np.mean(s4)) / np.mean(s2) ) *100
reportFile.write( 'Mean Control MAB phenos ->  Mean Injected MAB phenos' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %' + '\n' )


#-----------------
#------------------------
reportFile.close()


#%% Save Eye/ Body Ratio as CSV

# Control
s1a= Combined_Control_MAB_siblings_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s1b= Combined_Control_MAB_siblings_Body_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s1= s1a/s1b
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2a= Combined_Control_MAB_phenos_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s2b= Combined_Control_MAB_phenos_Body_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s2= s2a/s2b
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

# Injected
s3a= Combined_Injected_MAB_siblings_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s3b= Combined_Injected_MAB_siblings_Body_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s3= s3a/s3b
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))


s4a= Combined_Injected_MAB_phenos_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s4b= Combined_Injected_MAB_phenos_Body_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s4= s4a/s4b
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))


# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')

table = data_df1

table.to_csv('C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep' + 
             '/Eye_Body_Ratio.csv' , index=False)

#%% # Box Plot with Scatter all Eye to Body Ratio
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Control
s1a= Combined_Control_MAB_siblings_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s1b= Combined_Control_MAB_siblings_Body_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s1= s1a/s1b
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2a= Combined_Control_MAB_phenos_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s2b= Combined_Control_MAB_phenos_Body_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s2= s2a/s2b
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

# Injected
s3a= Combined_Injected_MAB_siblings_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s3b= Combined_Injected_MAB_siblings_Body_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s3= s3a/s3b
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))


s4a= Combined_Injected_MAB_phenos_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s4b= Combined_Injected_MAB_phenos_Body_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s4= s4a/s4b
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))


# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')

#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

vals[0]=vals[0][~np.isnan(vals[0])]
xs[0]= xs[0][0:len(vals[0])]

vals[1]=vals[1][~np.isnan(vals[1])]
xs[1]= xs[1][0:len(vals[1])]

vals[2]=vals[2][~np.isnan(vals[2])]
xs[2]= xs[2][0:len(vals[2])]

vals[3]=vals[3][~np.isnan(vals[3])]
xs[3]= xs[3][0:len(vals[3])]


plt.figure(1)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', 'black', 'black', ]
markers= ['.', '^', '.', '^']
sizes= [45, 20, 45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Eye Body Ratio')
    plt.tight_layout()
    plt.ylabel('Eye/ Body ')
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    
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
s1a= np.average([Control_MAB_siblings_Left_Head_Length, Control_MAB_siblings_Right_Head_Length], axis=0)
s1a= s1a * Body_conversion_3x
s1b= np.average([Control_MAB_siblings_Left_Body_Length, Control_MAB_siblings_Right_Body_Length], axis=0)
s1b= s1b * Body_conversion_3x

s2a= np.average([Control_MAB_phenos_Left_Head_Length, Control_MAB_phenos_Right_Head_Length], axis=0)
s2a= s2a * Body_conversion_3x
s2b= np.average([Control_MAB_phenos_Left_Body_Length, Control_MAB_phenos_Right_Body_Length], axis=0)
s2b= s2b * Body_conversion_3x


# Injected
s3a= np.average([Injected_MAB_siblings_Left_Head_Length, Injected_MAB_siblings_Right_Head_Length], axis=0)
s3a= s3a * Body_conversion_3x
s3b= np.average([Injected_MAB_siblings_Left_Body_Length, Injected_MAB_siblings_Right_Body_Length], axis=0)
s3b= s3b * Body_conversion_3x

s4a= np.average([Injected_MAB_phenos_Left_Head_Length, Injected_MAB_phenos_Right_Head_Length], axis=0)
s4a= s4a * Body_conversion_3x
s4b= np.average([Injected_MAB_phenos_Left_Body_Length, Injected_MAB_phenos_Right_Body_Length], axis=0)
s4b= s4b * Body_conversion_3x


s1= s1a/s1b
s2= s2a/s2b
s3= s3a/s3b
s4= s4a/s4b



reportFile.write('Eye Body Ratio: Control vs Injected' + '\n' )  
reportFile.write( '#---------------------------' + '\n' )
reportFile.write( '#---------------------------' + '\n' + '\n')

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)

reportFile.write('Mean Control MAB siblings = '      + str(s1_mean) + ' um' +'\n')
reportFile.write('Mean Control MAB phenos= '         + str(s2_mean) +' um' + '\n')
reportFile.write('Mean Injected MAB siblings = '     + str(s3_mean) +' um' + '\n')
reportFile.write('Mean Injected MAB phenos = '       + str(s4_mean) + ' um' +'\n' + '\n')

reportFile.write('#------------------------' + '\n')

## --------------
## Mean Control MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s3, equal_var = False)
reportFile.write( 'Means: Control MAB siblings vs Injected MAB siblings' + 
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
t, p = sp.stats.ttest_ind(s2, s4, equal_var = False)
reportFile.write( 'Means: Control MAB phenos vs Injected MAB phenos' + 
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
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
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
t, p = sp.stats.ttest_ind(s3, s4, equal_var = False)
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
percentage_change= ((np.mean(s1) - np.mean(s3)) / np.mean(s1) ) *100
reportFile.write( 'Mean Control MAB siblings ->  Mean Injected MAB sibllings' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %'+  '\n' + '\n' )

## Mean Control MAB pheno vs Mean Injected MAB phenos
percentage_change= ((np.mean(s2) - np.mean(s4)) / np.mean(s2) ) *100
reportFile.write( 'Mean Control MAB phenos ->  Mean Injected MAB phenos' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %' + '\n' )


#-----------------
#------------------------
reportFile.close()


#%% Save Head/ Body Ratio as CSV

# Control
s1a= np.average([Control_MAB_siblings_Left_Head_Length, Control_MAB_siblings_Right_Head_Length], axis=0)
s1a= s1a * Body_conversion_3x
s1b= np.average([Control_MAB_siblings_Left_Body_Length, Control_MAB_siblings_Right_Body_Length], axis=0)
s1b= s1b * Body_conversion_3x

s2a= np.average([Control_MAB_phenos_Left_Head_Length, Control_MAB_phenos_Right_Head_Length], axis=0)
s2a= s2a * Body_conversion_3x
s2b= np.average([Control_MAB_phenos_Left_Body_Length, Control_MAB_phenos_Right_Body_Length], axis=0)
s2b= s2b * Body_conversion_3x


# Injected
s3a= np.average([Injected_MAB_siblings_Left_Head_Length, Injected_MAB_siblings_Right_Head_Length], axis=0)
s3a= s3a * Body_conversion_3x
s3b= np.average([Injected_MAB_siblings_Left_Body_Length, Injected_MAB_siblings_Right_Body_Length], axis=0)
s3b= s3b * Body_conversion_3x

s4a= np.average([Injected_MAB_phenos_Left_Head_Length, Injected_MAB_phenos_Right_Head_Length], axis=0)
s4a= s4a * Body_conversion_3x
s4b= np.average([Injected_MAB_phenos_Left_Body_Length, Injected_MAB_phenos_Right_Body_Length], axis=0)
s4b= s4b * Body_conversion_3x


s1= s1a/s1b
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= s2a/s2b
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

s3= s3a/s3b
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))

s4= s4a/s4b
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))

# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')

table = data_df1

table.to_csv('C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep' + 
             '/Head_Body_Ratio.csv' , index=False)



#%% # Box Plot with Scatter all Head to Body Ratio
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------


# Control
s1a= np.average([Control_MAB_siblings_Left_Head_Length, Control_MAB_siblings_Right_Head_Length], axis=0)
s1a= s1a * Body_conversion_3x
s1b= np.average([Control_MAB_siblings_Left_Body_Length, Control_MAB_siblings_Right_Body_Length], axis=0)
s1b= s1b * Body_conversion_3x

s2a= np.average([Control_MAB_phenos_Left_Head_Length, Control_MAB_phenos_Right_Head_Length], axis=0)
s2a= s2a * Body_conversion_3x
s2b= np.average([Control_MAB_phenos_Left_Body_Length, Control_MAB_phenos_Right_Body_Length], axis=0)
s2b= s2b * Body_conversion_3x


# Injected
s3a= np.average([Injected_MAB_siblings_Left_Head_Length, Injected_MAB_siblings_Right_Head_Length], axis=0)
s3a= s3a * Body_conversion_3x
s3b= np.average([Injected_MAB_siblings_Left_Body_Length, Injected_MAB_siblings_Right_Body_Length], axis=0)
s3b= s3b * Body_conversion_3x

s4a= np.average([Injected_MAB_phenos_Left_Head_Length, Injected_MAB_phenos_Right_Head_Length], axis=0)
s4a= s4a * Body_conversion_3x
s4b= np.average([Injected_MAB_phenos_Left_Body_Length, Injected_MAB_phenos_Right_Body_Length], axis=0)
s4b= s4b * Body_conversion_3x


s1= s1a/s1b
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2= s2a/s2b
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

s3= s3a/s3b
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))

s4= s4a/s4b
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))

# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')

#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

vals[0]=vals[0][~np.isnan(vals[0])]
xs[0]= xs[0][0:len(vals[0])]

vals[1]=vals[1][~np.isnan(vals[1])]
xs[1]= xs[1][0:len(vals[1])]

vals[2]=vals[2][~np.isnan(vals[2])]
xs[2]= xs[2][0:len(vals[2])]

vals[3]=vals[3][~np.isnan(vals[3])]
xs[3]= xs[3][0:len(vals[3])]


plt.figure(1)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', 'black', 'black', ]
markers= ['.', '^', '.', '^']
sizes= [45, 20, 45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Head Body Ratio')
    plt.tight_layout()
    plt.ylabel('Head/ Body ')
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    
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
# Eye Body Ratio all Control fish eyes vs all Injected fish eyes

# Control
s1a= Combined_Control_MAB_siblings_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s1b= Combined_Control_MAB_siblings_Head_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once

s2a= Combined_Control_MAB_phenos_eye_major_axis_length    * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s2b= Combined_Control_MAB_phenos_Head_Length              * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once


# Injected
s3a= Combined_Injected_MAB_siblings_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s3b= Combined_Injected_MAB_siblings_Head_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once

s4a= Combined_Injected_MAB_phenos_eye_major_axis_length    * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s4b= Combined_Injected_MAB_phenos_Head_Length              * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once


s1= s1a/s1b
s2= s2a/s2b
s3= s3a/s3b
s4= s4a/s4b


reportFile.write('Eye Head Ratio: Control vs Injected' + '\n' )  
reportFile.write( '#---------------------------' + '\n' )
reportFile.write( '#---------------------------' + '\n' + '\n')

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)

reportFile.write('Mean Control MAB siblings = '      + str(s1_mean) + ' um' +'\n')
reportFile.write('Mean Control MAB phenos= '         + str(s2_mean) +' um' + '\n')
reportFile.write('Mean Injected MAB siblings = '     + str(s3_mean) +' um' + '\n')
reportFile.write('Mean Injected MAB phenos = '       + str(s4_mean) + ' um' +'\n' + '\n')

reportFile.write('#------------------------' + '\n')

## --------------
## Mean Control MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s3, equal_var = False)
reportFile.write( 'Means: Control MAB siblings vs Injected MAB siblings' + 
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
t, p = sp.stats.ttest_ind(s2, s4, equal_var = False)
reportFile.write( 'Means: Control MAB phenos vs Injected MAB phenos' + 
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
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
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
t, p = sp.stats.ttest_ind(s3, s4, equal_var = False)
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
percentage_change= ((np.mean(s1) - np.mean(s3)) / np.mean(s1) ) *100
reportFile.write( 'Mean Control MAB siblings ->  Mean Injected MAB sibllings' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %'+  '\n' + '\n' )

## Mean Control MAB pheno vs Mean Injected MAB phenos
percentage_change= ((np.mean(s2) - np.mean(s4)) / np.mean(s2) ) *100
reportFile.write( 'Mean Control MAB phenos ->  Mean Injected MAB phenos' + '\n' +
    'Percentage Difference: ' + str(percentage_change) + ' %' + '\n' )


#-----------------
#------------------------
reportFile.close()


#%% Save Eye/ Head Ratio as CSV

# Control
s1a= Combined_Control_MAB_siblings_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s1b= Combined_Control_MAB_siblings_Head_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s1= s1a/s1b
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2a= Combined_Control_MAB_phenos_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s2b= Combined_Control_MAB_phenos_Head_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s2= s2a/s2b
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

# Injected
s3a= Combined_Injected_MAB_siblings_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s3b= Combined_Injected_MAB_siblings_Head_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s3= s3a/s3b
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))


s4a= Combined_Injected_MAB_phenos_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s4b= Combined_Injected_MAB_phenos_Head_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s4= s4a/s4b
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))


# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')

table = data_df1

table.to_csv('C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep' + 
             '/Eye_Head_Ratio.csv' , index=False)


#%% # Box Plot with Scatter all Eye to Head Ratio
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Control
s1a= Combined_Control_MAB_siblings_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s1b= Combined_Control_MAB_siblings_Head_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s1= s1a/s1b
s1= pd.DataFrame(s1)
s1.columns=['Control_MAB_siblings']
s1 = s1.set_index(np.arange(0,len(s1)))

s2a= Combined_Control_MAB_phenos_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s2b= Combined_Control_MAB_phenos_Head_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s2= s2a/s2b
s2= pd.DataFrame(s2)
s2.columns=['Control_MAB_phenos']
s2 = s2.set_index(np.arange(0,len(s2)))

# Injected
s3a= Combined_Injected_MAB_siblings_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s3b= Combined_Injected_MAB_siblings_Head_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s3= s3a/s3b
s3= pd.DataFrame(s3)
s3.columns=['Injected_MAB_siblings']
s3 = s3.set_index(np.arange(0,len(s3)))


s4a= Combined_Injected_MAB_phenos_eye_major_axis_length  * Eye_conversion_8x           #Finding average then average fo that is the same as finding the average once
s4b= Combined_Injected_MAB_phenos_Head_Length            * Body_conversion_3x          #Finding average then average fo that is the same as finding the average once
s4= s4a/s4b
s4= pd.DataFrame(s4)
s4.columns=['Injected_MAB_phenos']
s4 = s4.set_index(np.arange(0,len(s4)))


# Create dataframe for plots
data_df1= s1.join(s2, how='left')
data_df1= data_df1.join(s3, how='left')
data_df1= data_df1.join(s4, how='left')

#------------------------

vals, names, xs = [],[],[]
for i, col in enumerate(data_df1.columns):
    vals.append(data_df1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, data_df1[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

vals[0]=vals[0][~np.isnan(vals[0])]
xs[0]= xs[0][0:len(vals[0])]

vals[1]=vals[1][~np.isnan(vals[1])]
xs[1]= xs[1][0:len(vals[1])]

vals[2]=vals[2][~np.isnan(vals[2])]
xs[2]= xs[2][0:len(vals[2])]

vals[3]=vals[3][~np.isnan(vals[3])]
xs[3]= xs[3][0:len(vals[3])]


plt.figure(1)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.35)

palette = ['black', 'black', 'black', 'black', ]
markers= ['.', '^', '.', '^']
sizes= [45, 20, 45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Eye Head Ratio')
    plt.tight_layout()
    plt.ylabel('Eye/ Head ')
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/All_Eye_Head_Ratio-Boxplot_with_scatter', dpi=900)  

plt.show()  

