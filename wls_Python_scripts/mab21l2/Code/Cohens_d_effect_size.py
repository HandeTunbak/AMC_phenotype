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
    return d

# Function to calculate 95% confidence interval for Cohen's d for normally distributed data
def cohens_d_ci_normal(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    t_statistic = (mean1 - mean2) / (pooled_std * np.sqrt(1 / n1 + 1 / n2))
    df = n1 + n2 - 2
    se = pooled_std * np.sqrt(1 / n1 + 1 / n2)
    ci = t.interval(0.95, df, loc=t_statistic, scale=se)  # 95% confidence interval
    return ci

# Function to calculate Cohen's d for non-normally distributed data
def cohens_d_non_normal(group1, group2):
    n1, n2 = len(group1), len(group2)
    u_statistic, p_value = mannwhitneyu(group1, group2)
    z = (u_statistic - (n1 * n2 / 2)) / sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
    return z * sqrt(1 / n1 + 1 / n2)

# Function to calculate 95% confidence interval for Cohen's d for non-normally distributed data
def cohens_d_ci_non_normal(group1, group2):
    n1, n2 = len(group1), len(group2)
    u_statistic, p_value = mannwhitneyu(group1, group2)
    z = (u_statistic - (n1 * n2 / 2)) / sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
    se = sqrt((n1 * n2 / 12))
    ci = (z - 1.96 * se, z + 1.96 * se)  # 95% confidence interval
    return ci, z


# Load data
input_file_path = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep/'

x = 3

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

# Calculate Cohen's d and 95% confidence interval for both normally and not normally distributed data
if p_value1 > 0.05 and p_value2 > 0.05:
    d1 = cohens_d_normal(group1, group2)
    ci1 = cohens_d_ci_normal(group1, group2)
    test1 = 'ind_t_test'
    df1 = len(group1) + len(group2) - 2
    p_val1 = 2 * (1 - norm.cdf(abs(d1)))
    u_statistic1 = None
else:
    d1 = cohens_d_non_normal(group1, group2)
    ci1 = cohens_d_ci_non_normal(group1, group2)
    test1 = 'mann_whitney_u_test'
    u_statistic1, p_val1 = mannwhitneyu(group1, group2)
    df1 = None

if p_value3 > 0.05 and p_value4 > 0.05:
    d2 = cohens_d_normal(group3, group4)
    ci2 = cohens_d_ci_normal(group3, group4)
    test2 = 'ind_t_test'
    df2 = len(group3) + len(group4) - 2
    p_val2 = 2 * (1 - norm.cdf(abs(d2)))
    u_statistic2 = None
else:
    d2 = cohens_d_non_normal(group3, group4)
    ci2 = cohens_d_ci_non_normal(group3, group4)
    test2 = 'mann_whitney_u_test'
    u_statistic2, p_val2 = mannwhitneyu(group3, group4)
    df2 = None

# Create summary table
summary_table = pd.DataFrame({
    'Group Pair': ['Control_MAB_siblings vs Injected_MAB_siblings', 'Control_MAB_phenos vs Injected_MAB_phenos'],
    "Cohen's d": [d1, d2],
    '95% CI': [ci1, ci2],
    'Statistical Test Used': [test1, test2],
    'Degrees of Freedom': [df1, df2],
    'U_statsitic': [u_statistic1,u_statistic2],
    'p-value': [p_val1, p_val2]
    
})

print(summary_table)
