# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:31:38 2024

This script compares effects sizes 

@author: hande
"""

#%%


import numpy as np
import scipy.stats as stats
import pandas as pd

#%%

def cohen_d(group1, group2):
    """Calculate Cohen's d for independent samples."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * std1*2 + (n2 - 1) * std2*2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / pooled_std
    return d

def standard_error_of_d(n1, n2, d):
    """Calculate the standard error of Cohen's d."""
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + (d**2 / (2 * (n1 + n2))))
    return se_d

def compare_effect_sizes(group1, group2, group3, group4):
    """Compare the effect sizes of two group pairs."""
    # Calculate Cohen's d for both comparisons
    d1 = cohen_d(group1, group2)
    d2 = cohen_d(group3, group4)
    
    # Calculate the standard errors of both effect sizes
    se_d1 = standard_error_of_d(len(group1), len(group2), d1)
    se_d2 = standard_error_of_d(len(group3), len(group4), d2)
    
    # Calculate the standard error of the difference between two effect sizes
    se_diff = np.sqrt(se_d1*2 + se_d2*2)
    
    # Calculate the z-score for the difference
    z = (d1 - d2) / se_diff
    
    # Calculate the p-value
    p_value = stats.norm.sf(abs(z)) * 2  # two-tailed test
    
    return d1, d2, z, p_value

#%%

input_file_path = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep/'

x= 8

x=x-1

if x==1:
    input_file_name= 'Average_Body_Length_(um).csv'
    
elif x==2: 
    input_file_name= 'Eccentricity.csv'
    
elif x==3:  
    input_file_name= 'Eye_Area_(um^2).csv'
    
elif x==4:  
    input_file_name= 'Eye_Body_Ratio.csv'
    
elif x==5:   
    input_file_name= 'Eye_Head_Ratio.csv'
    
elif x==6:     
    input_file_name= 'Eye_Lengths_(um).csv'
    
elif x==7:   
    input_file_name= 'Head_Body_Ratio.csv'
else: 
    print('error') 


input_file= str(input_file_path + input_file_name)

file= pd.read_csv(input_file)



#%%
# Example usage
group1 = file['Control_MAB_siblings'] # Replace with your data
group1= group1.dropna()

group2 = file['Injected_MAB_siblings']  # Replace with your data
group2= group2.dropna()

group3 = file['Control_MAB_phenos']  # Replace with your data
group3= group3.dropna()

group4 = file['Injected_MAB_phenos']  # Replace with your data
group4= group4.dropna()


# Compare effect sizes
d1, d2, z, p_value = compare_effect_sizes(group1, group2, group3, group4)
print(f"Effect size 1 (Cohen's d): {d1:.3f}")
print(f"Effect size 2 (Cohen's d): {d2:.3f}")
print(f"Z-score for the difference: {z:.3f}")
print(f"P-value: {p_value:.3f}")