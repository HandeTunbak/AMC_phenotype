# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Dr Hande Tunbak)
"""

from scipy.stats import shapiro, mannwhitneyu, t, norm
import pandas as pd
import numpy as np
from math import sqrt

# Function to calculate Cohen's d for normally distributed data
def cohens_d_normal(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / pooled_std
    d=d*-1
    
    return d

# Function to calculate 95% confidence interval for Cohen's d for normally distributed data
def cohens_d_ci_normal(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / pooled_std
    
    # Calculate the standard error of Cohen's d
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    
    # Calculate the 95% confidence interval for Cohen's d
    alpha = 0.05
    z = norm.ppf(1 - alpha/2)
    ci_lower = d - z * se_d
    ci_upper = d + z * se_d
    
    ci_lower = ci_lower *-1
    ci_upper = ci_upper *-1
    
    # if mean1> mean2:
    #     ci_lower = ci_lower *-1
    #     ci_upper = ci_upper *-1
    # else:
    #     pass
        
    return (ci_lower, ci_upper)

# Load data
input_file_path = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep/'

x = 8

x = x - 1

if x == 1:
    input_file_name = 'Average_Body_Length_(um).csv'
elif x == 2:
    input_file_name = 'Eccentricity.csv'
elif x == 3:
    input_file_name = 'Eye_Area_(um^2).csv'
elif x == 4:
    input_file_name = 'Eye_Body_Ratio.csv'
elif x == 5:
    input_file_name = 'Eye_Head_Ratio.csv'
elif x == 6:
    input_file_name = 'Eye_Lengths_(um).csv'
elif x == 7:
    input_file_name = 'Head_Body_Ratio.csv'
else:
    print('error')

input_file = str(input_file_path + input_file_name)

file = pd.read_csv(input_file)

group1 = file['Control_MAB_siblings'].dropna()
group2 = file['Injected_MAB_siblings'].dropna()
group3 = file['Control_MAB_phenos'].dropna()
group4 = file['Injected_MAB_phenos'].dropna()

# Check for normality using Shapiro-Wilk test
statistic1, p_value1 = shapiro(group1)
statistic2, p_value2 = shapiro(group2)
statistic3, p_value3 = shapiro(group3)
statistic4, p_value4 = shapiro(group4)

# Calculate Cohen's d and 95% confidence interval for normally distributed data
d1 = cohens_d_normal(group1, group2)
ci1 = cohens_d_ci_normal(group1, group2)
test1 = 'ind_t_test'
df1 = len(group1) + len(group2) - 2
p_val1 = 2 * (1 - norm.cdf(abs(d1)))
u_statistic1 = None

d2= cohens_d_normal(group3, group4)
ci2 = cohens_d_ci_normal(group3, group4)
test2 = 'ind_t_test'
df2 = len(group3) + len(group4) - 2
p_val2 = 2 * (1 - norm.cdf(abs(d2)))
u_statistic2 = None

# Create summary table
summary_table = pd.DataFrame({
    'Group Pair': ['Control_MAB_siblings vs Injected_MAB_siblings', 'Control_MAB_phenos vs Injected_MAB_phenos'],
    "Cohen's d (Control to Injected)": [d1, d2],
    '95% CI': [ci1, ci2],
    'Statistical Test Used': [test1, test2],
    'Degrees of Freedom': [df1, df2],
    'p-value': [p_val1, p_val2]
})
    
    
print(summary_table)
