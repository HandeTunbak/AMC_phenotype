# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Dr Hande Tunbak)
"""

'''If fisrt time using code, install dabest via pip:
    pip install dabest'''

#%%
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy
import pandas as pd
import scipy.stats as stats

#%% # Set paths for csv files  
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Location with subdirectories
Experiment_folder = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_TL/Testing_area/Effect_size_prep'

eye_data_df= pd.read_csv(Experiment_folder +  '/Eye_length_(um).csv') 
body_data_df= pd.read_csv(Experiment_folder +  '/Body_length_(um).csv') 

#%%

# Control group data
Control_eye_size = eye_data_df['Control']
Control_eye_size = Control_eye_size.dropna()
Control_body_size = body_data_df['Control']
Control_body_size = Control_body_size.dropna()


# Injected group data
Injected_eye_size = eye_data_df['Injected']
Injected_eye_size = Injected_eye_size.dropna()
Injected_body_size = body_data_df['Injected']
Injected_body_size = Injected_body_size.dropna()

# Calculate mean sizes for Control group
mean_Control_body_size = np.mean(Control_body_size, axis=0)
mean_Control_eye_size = np.mean(Control_eye_size, axis=0)

# Calculate percentage reductions using mean values for Control group
percentage_body_size_reduction = ((mean_Control_body_size - Injected_body_size) / mean_Control_body_size) * 100
percentage_eye_size_reduction = ((mean_Control_eye_size - Injected_eye_size) / mean_Control_eye_size) * 100

# Create box plots
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.boxplot([percentage_body_size_reduction, percentage_eye_size_reduction], labels=['Body Size', 'Eye Size'], patch_artist=True, showmeans=True, boxprops=dict(facecolor="lightblue"))
plt.title('Percentage Reductions')
plt.ylabel('Percentage Reduction')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)  # Add horizontal line at y=0

# Add annotation to indicate which group had greater reduction
if np.mean(percentage_body_size_reduction) > np.mean(percentage_eye_size_reduction):
    greater_reduction = 'Body Size'
else:
    greater_reduction = 'Eye Size'

plt.text(1.05, np.mean([percentage_body_size_reduction, percentage_eye_size_reduction]) - 2.5,
         f'{greater_reduction} had greater reduction', fontsize=10, color='red')

plt.subplot(1, 2, 2)
plt.hist([percentage_body_size_reduction, percentage_eye_size_reduction], bins=10, label=['Body Size', 'Eye Size'], color=['lightblue', 'lightgreen'], alpha=0.7)
plt.title('Histogram of Percentage Reductions')
plt.xlabel('Percentage Reduction')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)  # Add vertical line at x=0

plt.tight_layout()
plt.show()


# Check for skewness
body_size_skewness = stats.skew(percentage_body_size_reduction)
eye_size_skewness = stats.skew(percentage_eye_size_reduction)

# Choose appropriate test based on skewness
if abs(body_size_skewness) < 0.5 and abs(eye_size_skewness) < 0.5:
    # If both distributions are approximately symmetric, perform paired t-test
    test_statistic, p_value = stats.ttest_rel(percentage_body_size_reduction, percentage_eye_size_reduction)
    test= 'Paired_t_test'
else:
    # If either distribution is skewed, perform Wilcoxon signed-rank test
    test_statistic, p_value = stats.wilcoxon(percentage_body_size_reduction, percentage_eye_size_reduction)
    test= 'Wilcoxon_Rank_Sign_test'

print(f"Test results: statistic = {test_statistic:.2f}, p-value = {p_value:.4f}")

Summary_stats= pd.DataFrame([[test, test_statistic, p_value]], columns=['Test', 'test_statistic', 'p_value'])


