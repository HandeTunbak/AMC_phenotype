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
Experiment_folder = 'C:/Users/Hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep'

eye_data_df= pd.read_csv(Experiment_folder +  '/Eye_Lengths_(um).csv') 
body_data_df= pd.read_csv(Experiment_folder +  '/Body_Length_(um).csv') 

#%%

# Control group data
Control_MAB_sib_eye_size = eye_data_df['Control_MAB_siblings']
Control_MAB_sib_eye_size = Control_MAB_sib_eye_size.dropna()
Control_MAB_sib_body_size = body_data_df['Control_MAB_siblings']
Control_MAB_sib_body_size = Control_MAB_sib_body_size.dropna()

Control_MAB_pheno_eye_size = eye_data_df['Control_MAB_phenos']
Control_MAB_pheno_eye_size = Control_MAB_pheno_eye_size.dropna()
Control_MAB_pheno_body_size = body_data_df['Control_MAB_phenos']
Control_MAB_pheno_body_size = Control_MAB_pheno_body_size.dropna()


# Injected group data
Injected_MAB_sib_eye_size = eye_data_df['Injected_MAB_siblings']
Injected_MAB_sib_eye_size = Injected_MAB_sib_eye_size.dropna()
Injected_MAB_sib_body_size = body_data_df['Injected_MAB_siblings']
Injected_MAB_sib_body_size = Injected_MAB_sib_body_size.dropna()

Injected_MAB_pheno_eye_size = eye_data_df['Injected_MAB_phenos']
Injected_MAB_pheno_eye_size = Injected_MAB_pheno_eye_size.dropna()
Injected_MAB_pheno_body_size = body_data_df['Injected_MAB_phenos']
Injected_MAB_pheno_body_size = Injected_MAB_pheno_body_size.dropna()


# Calculate mean sizes for Control group
mean_Control_MAB_sib_eye_size = np.mean(Control_MAB_sib_eye_size, axis=0)
mean_Control_MAB_sib_body_size = np.mean(Control_MAB_sib_body_size, axis=0)

mean_Control_MAB_pheno_eye_size = np.mean(Control_MAB_pheno_eye_size, axis=0)
mean_Control_MAB_pheno_body_size = np.mean(Control_MAB_pheno_body_size, axis=0)


# Calculate percentage reductions using mean values for Control group
percentage_MAB_sib_eye_size_reduction = ((mean_Control_MAB_sib_eye_size - Injected_MAB_sib_eye_size) / mean_Control_MAB_sib_eye_size) * 100
percentage_MAB_sib_body_size_reduction = ((mean_Control_MAB_sib_body_size - Injected_MAB_sib_body_size) / mean_Control_MAB_sib_body_size) * 100


percentage_MAB_pheno_eye_size_reduction = ((mean_Control_MAB_pheno_eye_size - Injected_MAB_pheno_eye_size) / mean_Control_MAB_pheno_eye_size) * 100
percentage_MAB_pheno_body_size_reduction = ((mean_Control_MAB_pheno_body_size - Injected_MAB_pheno_body_size) / mean_Control_MAB_pheno_body_size) * 100


#%% delta Siblings
#-----------------------------

# Create box plots
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.boxplot([percentage_MAB_sib_body_size_reduction, percentage_MAB_sib_eye_size_reduction], labels=['Body Size', 'Eye Size'], patch_artist=True, showmeans=True, boxprops=dict(facecolor="lightblue"))
plt.title('Percentage Reductions Sibs')
plt.ylabel('Percentage Reduction')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)  # Add horizontal line at y=0
plt.ylim(0,18)

# Add annotation to indicate which group had greater reduction
if np.mean(percentage_MAB_sib_body_size_reduction) > np.mean(percentage_MAB_sib_eye_size_reduction):
    greater_reduction = 'Body Size'
else:
    greater_reduction = 'Eye Size'

plt.text(1.05, np.mean([percentage_MAB_sib_body_size_reduction, percentage_MAB_sib_eye_size_reduction]) - 2.5,
         f'{greater_reduction} had greater reduction', fontsize=10, color='red')

plt.subplot(1, 2, 2)
plt.hist([percentage_MAB_sib_body_size_reduction, percentage_MAB_sib_eye_size_reduction], bins=10, label=['Body Size', 'Eye Size'], color=['lightblue', 'lightgreen'], alpha=0.7)
plt.title('Histogram of Percentage Reductions')
plt.xlabel('Percentage Reduction')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)  # Add vertical line at x=0
plt.ylim(0,9)

plt.tight_layout()
plt.show()


# Check for skewness
body_size_skewness = stats.skew(percentage_MAB_sib_body_size_reduction)
eye_size_skewness = stats.skew(percentage_MAB_sib_eye_size_reduction)

# Choose appropriate test based on skewness
if abs(body_size_skewness) < 0.5 and abs(eye_size_skewness) < 0.5:
    # If both distributions are approximately symmetric, perform paired t-test
    test_statistic, p_value = stats.ttest_rel(percentage_MAB_sib_body_size_reduction, percentage_MAB_sib_eye_size_reduction)
    test= 'Paired_t_test'
else:
    # If either distribution is skewed, perform Wilcoxon signed-rank test
    test_statistic, p_value = stats.wilcoxon(percentage_MAB_sib_body_size_reduction, percentage_MAB_sib_eye_size_reduction)
    test= 'Wilcoxon_Rank_Sign_test'

print(f"Test results: statistic = {test_statistic:.2f}, p-value = {p_value:.4f}")

Summary_stats_sibs= pd.DataFrame([[test, test_statistic, p_value]], columns=['Test', 'test_statistic', 'p_value'])

#%% Delta phenos
#-----------------------------

# Create box plots
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.boxplot([percentage_MAB_pheno_body_size_reduction, percentage_MAB_pheno_eye_size_reduction], labels=['Body Size', 'Eye Size'], patch_artist=True, showmeans=True, boxprops=dict(facecolor="lightblue"))
plt.title('Percentage Reductions Phenos')
plt.ylabel('Percentage Reduction')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)  # Add horizontal line at y=0
plt.ylim(0,18)

# Add annotation to indicate which group had greater reduction
if np.mean(percentage_MAB_pheno_body_size_reduction) > np.mean(percentage_MAB_pheno_eye_size_reduction):
    greater_reduction = 'Body Size'
else:
    greater_reduction = 'Eye Size'

plt.text(1.05, np.mean([percentage_MAB_pheno_body_size_reduction, percentage_MAB_pheno_eye_size_reduction]) - 2.5,
         f'{greater_reduction} had greater reduction', fontsize=10, color='red')

plt.subplot(1, 2, 2)
plt.hist([percentage_MAB_pheno_body_size_reduction, percentage_MAB_pheno_eye_size_reduction], bins=10, label=['Body Size', 'Eye Size'], color=['lightblue', 'lightgreen'], alpha=0.7)
plt.title('Histogram of Percentage Reductions')
plt.xlabel('Percentage Reduction')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)  # Add vertical line at x=0
plt.ylim(0,9)

plt.tight_layout()
plt.show()


# Check for skewness
body_size_skewness = stats.skew(percentage_MAB_pheno_body_size_reduction)
eye_size_skewness = stats.skew(percentage_MAB_pheno_eye_size_reduction)

# Choose appropriate test based on skewness
if abs(body_size_skewness) < 0.5 and abs(eye_size_skewness) < 0.5:
    # If both distributions are approximately symmetric, perform paired t-test
    test_statistic, p_value = stats.ttest_rel(percentage_MAB_pheno_body_size_reduction, percentage_MAB_pheno_eye_size_reduction)
    test= 'Paired_t_test'
else:
    # If either distribution is skewed, perform Wilcoxon signed-rank test
    test_statistic, p_value = stats.wilcoxon(percentage_MAB_pheno_body_size_reduction, percentage_MAB_pheno_eye_size_reduction)
    test= 'Wilcoxon_Rank_Sign_test'

print(f"Test results: statistic = {test_statistic:.2f}, p-value = {p_value:.4f}")

Summary_stats_pheno= pd.DataFrame([[test, test_statistic, p_value]], columns=['Test', 'test_statistic', 'p_value'])


#%% Comparing % reduction for body length and eye lengths for each parameter

s1=percentage_MAB_pheno_body_size_reduction
s2=percentage_MAB_sib_body_size_reduction

# Check for skewness
body_size_skewness1 = stats.skew(s1)
body_size_skewness2 = stats.skew(s2)

# Choose appropriate test based on skewness
if abs(body_size_skewness1) < 0.5 and abs(body_size_skewness2) < 0.5:
    test_statistic, p_value = stats.ttest_ind(s1, s2)
    test = 'Unpaired_t_test'

else:
    # If either distribution is skewed, perform Wilcoxon signed-rank test
    test_statistic, p_value = stats.mannwhitneyu(s1, s2)
    test= 'Mann_Whitney_U_test'


Summary_percent_body_reduction= pd.DataFrame([[test, test_statistic, p_value]], columns=['Test', 'test_statistic', 'p_value'])
mean_sib_body_percent_reduction = np.mean(s1)
mean_pheno_body_percent_reduction = np.mean(s2)


#------------------------

s1=percentage_MAB_pheno_eye_size_reduction
s2=percentage_MAB_sib_eye_size_reduction

# Check for skewness
eye_size_skewness1 = stats.skew(s1)
eye_size_skewness2 = stats.skew(s2)

# Choose appropriate test based on skewness
if abs(eye_size_skewness1) < 0.5 and abs(eye_size_skewness2) < 0.5:
    test_statistic, p_value = stats.ttest_ind(s1, s2)
    test = 'Unpaired_t_test'

else:
    # If either distribution is skewed, perform Wilcoxon signed-rank test
    test_statistic, p_value = stats.mannwhitneyu(s1, s2)
    test= 'Mann_Whitney_U_test'


Summary_percent_eye_reduction= pd.DataFrame([[test, test_statistic, p_value]], columns=['Test', 'test_statistic', 'p_value'])
mean_sib_eye_percent_reduction = np.mean(s1)
mean_pheno_eye_percent_reduction = np.mean(s2)









