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
Experiment_folder = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_TL_Brains_and_Bodies/Testing_area/Brain_images/'

#------------------------------
# # Control
Control_brain_paramaters= pd.read_csv(Experiment_folder + 
                                               'Control' + '/Control_Brain_Parameters.csv') 

Injected_brain_paramaters= pd.read_csv(Experiment_folder + 
                                               'Injected' + '/Injected_Brain_Parameters.csv')   

#%% Set conversion to microns for plotting and stats
# 92 pixels = 166.06 microns according to scale on Fiji which is defined by the confocal imaging settings and lens etc.
single_pixel_length = 166.92/92 # microns
# single_micron_length = 92/166.92 # pixels 

# Control
Control_brain_paramaters['Forebrain_Lengths']   = Control_brain_paramaters['Forebrain_Lengths']   * single_pixel_length
Control_brain_paramaters['Midbrain_Lengths']    = Control_brain_paramaters['Midbrain_Lengths']    * single_pixel_length
Control_brain_paramaters['Hindbrain_Lengths']   = Control_brain_paramaters['Hindbrain_Lengths']   * single_pixel_length
Control_brain_paramaters['Whole_Brain_Lengths'] = Control_brain_paramaters['Whole_Brain_Lengths'] * single_pixel_length

Control_brain_paramaters['Forebrain_Widths']    = Control_brain_paramaters['Forebrain_Widths']    * single_pixel_length
Control_brain_paramaters['Midbrain_Widths']     = Control_brain_paramaters['Midbrain_Widths']     * single_pixel_length
Control_brain_paramaters['Hindbrain_Widths']    = Control_brain_paramaters['Hindbrain_Widths']    * single_pixel_length

# Injected
Injected_brain_paramaters['Forebrain_Lengths']   = Injected_brain_paramaters['Forebrain_Lengths']   * single_pixel_length
Injected_brain_paramaters['Midbrain_Lengths']    = Injected_brain_paramaters['Midbrain_Lengths']    * single_pixel_length
Injected_brain_paramaters['Hindbrain_Lengths']   = Injected_brain_paramaters['Hindbrain_Lengths']   * single_pixel_length
Injected_brain_paramaters['Whole_Brain_Lengths'] = Injected_brain_paramaters['Whole_Brain_Lengths'] * single_pixel_length

Injected_brain_paramaters['Forebrain_Widths']    = Injected_brain_paramaters['Forebrain_Widths']    * single_pixel_length
Injected_brain_paramaters['Midbrain_Widths']     = Injected_brain_paramaters['Midbrain_Widths']     * single_pixel_length
Injected_brain_paramaters['Hindbrain_Widths']    = Injected_brain_paramaters['Hindbrain_Widths']    * single_pixel_length

#%%

# Control
Control_forebrain_whole_ratio = Control_brain_paramaters['Forebrain_Widths'] / Control_brain_paramaters['Whole_Brain_Lengths']
Control_midbrain_whole_ratio = Control_brain_paramaters['Midbrain_Widths'] / Control_brain_paramaters['Whole_Brain_Lengths']
Control_hindbrain_whole_ratio = Control_brain_paramaters['Hindbrain_Widths'] / Control_brain_paramaters['Whole_Brain_Lengths']

Control_forebrain_self_ratio = Control_brain_paramaters['Forebrain_Widths'] / Control_brain_paramaters['Forebrain_Lengths']
Control_midbrain_self_ratio = Control_brain_paramaters['Midbrain_Widths'] / Control_brain_paramaters['Midbrain_Lengths']
Control_hindbrain_self_ratio = Control_brain_paramaters['Hindbrain_Widths'] / Control_brain_paramaters['Hindbrain_Lengths']

# Injected
Injected_forebrain_whole_ratio = Injected_brain_paramaters['Forebrain_Widths'] / Injected_brain_paramaters['Whole_Brain_Lengths']
Injected_midbrain_whole_ratio = Injected_brain_paramaters['Midbrain_Widths'] / Injected_brain_paramaters['Whole_Brain_Lengths']
Injected_hindbrain_whole_ratio = Injected_brain_paramaters['Hindbrain_Widths'] / Injected_brain_paramaters['Whole_Brain_Lengths']

Injected_forebrain_self_ratio = Injected_brain_paramaters['Forebrain_Widths'] / Injected_brain_paramaters['Forebrain_Lengths']
Injected_midbrain_self_ratio = Injected_brain_paramaters['Midbrain_Widths'] / Injected_brain_paramaters['Midbrain_Lengths']
Injected_hindbrain_self_ratio = Injected_brain_paramaters['Hindbrain_Widths'] / Injected_brain_paramaters['Hindbrain_Lengths']


#%% Do Stats for Forebrain

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Forebrain_Ratio_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Forebrain Whole Brain Length Ratio
s1= Control_forebrain_whole_ratio
s2= Injected_forebrain_whole_ratio
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Forebrain Width Whole-brain Length Ratio: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change))

#-----------------------------------------
reportFile.write( '\n' + '#---------------------------')
reportFile.write ('\n' )
reportFile.write( '\n' + '#----------------------------------------------' + '\n' )
#-----------------------------------------

# Forebrain self Lenth Ratio
s1= Control_forebrain_self_ratio
s2= Injected_forebrain_self_ratio

reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Forebrain Width Self Length Ratio: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change))

#------------------------
reportFile.close()



#%% Do Stats for Midbrain

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Midbrain_Ratio_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Forebrain Whole Brain Length Ratio
s1= Control_midbrain_whole_ratio
s2= Injected_midbrain_whole_ratio
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Midbrain Width Whole-brain Length Ratio: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change))

#-----------------------------------------
reportFile.write( '\n' + '#---------------------------')
reportFile.write ('\n' )
reportFile.write( '\n' + '#----------------------------------------------' + '\n' )
#-----------------------------------------

# Forebrain self Lenth Ratio
s1= Control_midbrain_self_ratio
s2= Injected_midbrain_self_ratio

reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Midbrain Width Self Length Ratio: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change))

#------------------------
reportFile.close()



#%% Do Stats for Hindbrain

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Hindbrain_Ratio_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Forebrain Whole Brain Length Ratio
s1= Control_hindbrain_whole_ratio
s2= Injected_hindbrain_whole_ratio
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Hindbrain Width Whole-brain Length Ratio: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change))

#-----------------------------------------
reportFile.write( '\n' + '#---------------------------')
reportFile.write ('\n' )
reportFile.write( '\n' + '#----------------------------------------------' + '\n' )
#-----------------------------------------

# Forebrain self Lenth Ratio
s1= Control_hindbrain_self_ratio
s2= Injected_hindbrain_self_ratio

reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Hindbrain Width Self Length Ratio: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change))

#------------------------
reportFile.close()

#%% # Box Plot with Scatter Average Body Lengths 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Control 
s1a= Control_forebrain_self_ratio  
s1a= pd.DataFrame([s1a]).T
s1a.columns=['Control forebrain']
s1a = s1a.set_index(np.arange(0,len(s1a)))

s2a= Control_midbrain_self_ratio
s2a= pd.DataFrame([s2a]).T
s2a.columns=['Control midbrain']
s2a = s2a.set_index(np.arange(0,len(s2a)))

s3a= Control_hindbrain_self_ratio
s3a= pd.DataFrame([s3a]).T
s3a.columns=['Control hindbrain']
s3a = s3a.set_index(np.arange(0,len(s3a)))


# Injected
s1b= Injected_forebrain_self_ratio  
s1b= pd.DataFrame([s1b]).T
s1b.columns=['Injected forebrain']
s1b = s1b.set_index(np.arange(0,len(s1b)))

s2b= Injected_midbrain_self_ratio
s2b= pd.DataFrame([s2b]).T
s2b.columns=['Injected midbrain']
s2b = s2b.set_index(np.arange(0,len(s2b)))

s3b= Injected_hindbrain_self_ratio
s3b= pd.DataFrame([s3b]).T
s3b.columns=['Injected hindbrain']
s3b = s3b.set_index(np.arange(0,len(s3b)))


# Create dataframe for plots
data_df1= s1a.join(s1b, how='left')
data_df1= data_df1.join(s2a, how='left')
data_df1= data_df1.join(s2b, how='left')
data_df1= data_df1.join(s3a, how='left')
data_df1= data_df1.join(s3b, how='left')


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

vals[4]=vals[4][~np.isnan(vals[4])]
xs[4]= xs[4][0:len(vals[4])]

vals[5]=vals[5][~np.isnan(vals[5])]
xs[5]= xs[5][0:len(vals[5])]


plt.figure(3)
  
plt.boxplot(vals, labels=names, showfliers=False,   widths=0.40)

palette = ['black', 'black', 'black', 'black', 'black', 'black',]
markers= ['.', '^', '.', '^', '.', '^']
sizes= [45, 20, 45, 20, 45, 20]

for x, val, c, mi, sizing in zip(xs, vals, palette, markers, sizes):
    plt.scatter(x, val, alpha=0.4, color=c, marker=mi, s=sizing)
    plt.title('Brain Ratios')
    plt.tight_layout()
    ax= plt.subplot()

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Self_Brain_Ratios-Boxplot_with_scatter', dpi=1200) 
 
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Self_Brain_Ratios-Boxplot_with_scatter.eps', dpi=1200)  

plt.show()    

#%% Do Stats for Forebrain Length and Width

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Forebrain_Lengths_Widths_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Forebrain Widths
s1= Control_brain_paramaters['Forebrain_Widths']
s2= Injected_brain_paramaters['Forebrain_Widths']
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Forebrain Width: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change))

#-----------------------------------------
reportFile.write( '\n' + '#---------------------------')
reportFile.write ('\n' )
reportFile.write( '\n' + '#----------------------------------------------' + '\n' )
#-----------------------------------------

#-----------
# Forebrain Lengths
s1= Control_brain_paramaters['Forebrain_Lengths']
s2= Injected_brain_paramaters['Forebrain_Lengths']
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Forebrain Lengths: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change))

#------------------------
reportFile.close()


#%% Do Stats for Midbrain Length and Width

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Midbrain_Lengths_Widths_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Midbrain Widths
s1= Control_brain_paramaters['Midbrain_Widths']
s2= Injected_brain_paramaters['Midbrain_Widths']
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Midbrain Width: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change))

#-----------------------------------------
reportFile.write( '\n' + '#---------------------------')
reportFile.write ('\n' )
reportFile.write( '\n' + '#----------------------------------------------' + '\n' )
#-----------------------------------------

#-----------
# Midbrain Lengths
s1= Control_brain_paramaters['Midbrain_Lengths']
s2= Injected_brain_paramaters['Midbrain_Lengths']
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Midbrain Lengths: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change))

#------------------------
reportFile.close()


#%% Do Stats for Hindbrain Length and Width

# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Hindbrain_Lengths_Widths_Control_vs_Injected.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Hindbrain Widths
s1= Control_brain_paramaters['Hindbrain_Widths']
s2= Injected_brain_paramaters['Hindbrain_Widths']
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Hindbrain Width: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change))

#-----------------------------------------
reportFile.write( '\n' + '#---------------------------')
reportFile.write ('\n' )
reportFile.write( '\n' + '#----------------------------------------------' + '\n' )
#-----------------------------------------

#-----------
# Hindbrain Lengths
s1= Control_brain_paramaters['Hindbrain_Lengths']
s2= Injected_brain_paramaters['Hindbrain_Lengths']
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('HindBrain Lengths: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
reportFile.write('Mean Uninjected = ' + str(s1_mean) + '\n')
reportFile.write('Mean Injected = '  + str(s2_mean) + '\n')

# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2, equal_var = False)
reportFile.write('\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n')

Percentage_change = ((np.mean(s1) - np.mean(s2)) / np.mean(s1)) *100
reportFile.write ('\n' + '\n' + 'Percentage change (((control- injected)/control)*100) is= ' + str(Percentage_change))

#------------------------
reportFile.close()

#%%


#%% Finish