# -*- coding: utf-8 -*-
"""
Created on %(date)s

If using code fro first time install pingouin
        $ pip install pingouin

@author: %(Dr Hande Tunbak)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg

#%%

input_file_path = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep/Inputs_csv_files/'

x = 5

x = x - 1

if x == 1:
    input_file_name = 'Input_Average_Body_Length_(um).csv'
elif x == 2:
    input_file_name = 'Input_Eccentricity.csv'
elif x == 3:
    input_file_name = 'Input_Eye_Area_(um^2).csv'
elif x == 4:
    input_file_name = 'Input_Eye_Body_Ratio.csv'
elif x == 5:
    input_file_name = 'Input_Eye_Head_Ratio.csv'
elif x == 6:
    input_file_name = 'Input_Eye_Lengths_(um).csv'
elif x == 7:
    input_file_name = 'Input_Head_Body_Ratio.csv'
else:
    print('error')

input_file = str(input_file_path + input_file_name)
file = pd.read_csv(input_file)

df = file.drop('Condition', axis=1)

#%%
# Clean up whitespace in Treatment column
df['Treatment'] = df['Treatment'].str.strip()

# Debugging: Check unique values before replacement
print("Unique values in 'Treatment' before replacement:", df['Treatment'].unique())

# Replace empty strings with 'Uninjected'
df['Treatment'].replace('', 'Uninjected', inplace=True)

# Replace 'wls-/-' with 'Injected'
df['Treatment'].replace('wls-/-', 'Injected', inplace=True)

# Debugging: Check unique values after replacement
print("Unique values in 'Treatment' after replacement:", df['Treatment'].unique())

# Replace 'mab-/-' with 'mab_mut' in the Genotype column
df['Genotype'].replace('mab-/-', 'mab_mut', inplace=True)

# Debugging: Check unique values in 'Genotype'
print("Unique values in 'Genotype':", df['Genotype'].unique())

#%%

def permutation_anova(df, dv, between, within, n_perm=1000):
    observed_f = calculate_observed_f(df, dv, between, within)
    permuted_fs = {'Genotype': [], 'Treatment': [], 'Interaction': []}  # Store permuted F-statistics for each factor
    for _ in range(n_perm):
        permuted_df = df.copy()
        permuted_df[within] = np.random.permutation(permuted_df[within])
        permuted_f = calculate_observed_f(permuted_df, dv, between, within)
        for factor in permuted_f:
            permuted_fs[factor].append(permuted_f[factor])
    p_values = {}
    for factor, permuted_f in permuted_fs.items():
        observed = observed_f[factor]
        # Count the number of permuted F-statistics that are greater than or equal to the observed F-statistic
        count = sum(perm >= observed for perm in permuted_f)
        # Calculate the p-value
        p_values[factor] = (count + 1) / (n_perm + 1)
    return observed_f, p_values

def calculate_observed_f(df, dv, between, within):
    # Calculate group means
    group_means = df.groupby([between, within])[dv].mean().unstack(level=1)  # Unstack by Treatment
    # Debugging: Print group means
    print("Group means:\n", group_means)

    # Calculate overall mean
    overall_mean = df[dv].mean()
    # Debugging: Print overall mean
    print("Overall mean:", overall_mean)

    # Calculate sum of squares for Genotype
    ss_between_genotype = ((group_means.loc['mab_sibs'] - overall_mean).pow(2).sum() +
                           (group_means.loc['mab_mut'] - overall_mean).pow(2).sum())
    ss_within_genotype = ((df.groupby(between)[dv].mean() - group_means.loc['mab_sibs']).pow(2).sum() + 
                         (df.groupby(between)[dv].mean() - group_means.loc['mab_mut']).pow(2).sum())

    # Debugging: Print sum of squares for Genotype
    print("SS between Genotype:", ss_between_genotype)
    print("SS within Genotype:", ss_within_genotype)

    # Ensure that ss_within_genotype is not zero
    if ss_within_genotype == 0:
        f_statistic_genotype = np.nan  # Set F-statistic to NaN if ss_within_genotype is zero
    else:
        f_statistic_genotype = (ss_between_genotype / len(df[between].unique())) / \
                               (ss_within_genotype / (len(df) - len(df[between].unique())))
    # Debugging: Print F-statistic for Genotype
    print("F-statistic Genotype:", f_statistic_genotype)

    # Calculate sum of squares for Treatment
    ss_between_treatment = ((group_means['Uninjected'] - overall_mean).pow(2).sum() +
                            (group_means['Injected'] - overall_mean).pow(2).sum())
    ss_within_treatment = ((df.groupby(within)[dv].mean() - group_means['Uninjected']).pow(2).sum() +
                          (df.groupby(within)[dv].mean() - group_means['Injected']).pow(2).sum())

    # Debugging: Print sum of squares for Treatment
    print("SS between Treatment:", ss_between_treatment)
    print("SS within Treatment:", ss_within_treatment)

    # Ensure that ss_within_treatment is not zero
    if ss_within_treatment == 0:
        f_statistic_treatment = np.nan  # Set F-statistic to NaN if ss_within_treatment is zero
    else:
        f_statistic_treatment = (ss_between_treatment / len(df[within].unique())) / \
                                (ss_within_treatment / (len(df) - len(df[within].unique())))
    # Debugging: Print F-statistic for Treatment
    print("F-statistic Treatment:", f_statistic_treatment)

    # Calculate interaction between Genotype and Treatment
    interaction_ss = ((df.groupby([between, within])[dv].mean().unstack(level=1) - group_means).pow(2).sum().sum())
    df_interaction = (len(df[between].unique()) - 1) * (len(df[within].unique()) - 1)

    # Debugging: Print interaction sum of squares
    print("Interaction SS:", interaction_ss)
    print("Degrees of freedom interaction:", df_interaction)

    if interaction_ss == 0:
        f_statistic_interaction = np.nan
    else:
        f_statistic_interaction = interaction_ss / df_interaction
    # Debugging: Print F-statistic for Interaction
    print("F-statistic Interaction:", f_statistic_interaction)

    return {'Genotype': f_statistic_genotype, 'Treatment': f_statistic_treatment, 'Interaction': f_statistic_interaction}

# Run permutation ANOVA
observed_f, p_values = permutation_anova(df, dv='Value', between='Genotype', within='Treatment', n_perm=1000)

# Extract effects of genotype and treatment separately
genotype_effect = observed_f['Genotype']
genotype_p_value = p_values['Genotype']
treatment_effect = observed_f['Treatment']
treatment_p_value = p_values['Treatment']
interaction_effect = observed_f['Interaction']
interaction_p_value = p_values['Interaction']

print("Genotype Effect:", genotype_effect)
print("Genotype p-value:", genotype_p_value)
print("Treatment Effect:", treatment_effect)
print("Treatment p-value:", treatment_p_value)
print("Interaction Effect:", interaction_effect)
print("Interaction p-value:", interaction_p_value)
