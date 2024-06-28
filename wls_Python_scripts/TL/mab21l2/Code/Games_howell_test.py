# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Dr Hande Tunbak)
"""

import pandas as pd
import numpy as np
import pingouin as pg
from statsmodels.formula.api import ols
import statsmodels.api as sm

#%%
input_file_path = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep/Inputs_csv_files/'

x = 3


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

df = file

# Debugging: Print first few rows to verify the data
print("First few rows of the DataFrame:")
print(df.head())

# Check for missing values and handle them (e.g., drop or impute)
print("Missing values in each column:")
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Check data types and ensure they are correct
print("Data types of columns:")
print(df.dtypes)

# Clean up whitespace in Treatment column
df['Treatment'] = df['Treatment'].str.strip()

# Replace empty strings with 'Uninjected'
df['Treatment'] = df['Treatment'].replace('', 'Uninjected')

# Replace 'wls-/-' with 'Injected'
df['Treatment'] = df['Treatment'].replace('wls-/-', 'Injected')

# Replace 'mab-/-' with 'mab_mut' in the Genotype column
df['Genotype'] = df['Genotype'].replace('mab-/-', 'mab_mut')

# Perform Two-Way ANOVA
anova_results = pg.anova(data=df, dv='Value', between=['Genotype', 'Treatment'], detailed=True)

# Print ANOVA results
print(anova_results)


#%% different lib to do the same thing
# Perform Two-Way ANOVA
model = ols('Value ~ C(Genotype) * C(Treatment)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=3)
print("Two-Way ANOVA Results:")
print(anova_table)

#



# Check assumptions (normality and homogeneity of variances)
shapiro_genotype = pg.normality(data=df, dv='Value', group='Genotype')
shapiro_treatment = pg.normality(data=df, dv='Value', group='Treatment')
levene = pg.homoscedasticity(data=df, dv='Value', group='Genotype')

# Print assumption check results
print("\nShapiro-Wilk Normality Test for Genotype:")
print(shapiro_genotype)
print("\nShapiro-Wilk Normality Test for Treatment:")
print(shapiro_treatment)
print("\nLevene's Test for Variance Homogeneity:")
print(levene)

# Extract p-values for the checks
p_interaction = anova_results.loc[anova_results['Source'] == 'Genotype * Treatment', 'p-unc'].values[0].item()
p_genotype = anova_results.loc[anova_results['Source'] == 'Genotype', 'p-unc'].values[0].item()
p_treatment = anova_results.loc[anova_results['Source'] == 'Treatment', 'p-unc'].values[0].item()

# Perform Games-Howell post-hoc test if there is a significant interaction effect
if p_interaction < 0.05:
    print("\nPerforming Games-Howell post-hoc test for interaction effect...")
    df['Interaction'] = df['Genotype'] + "_" + df['Treatment']
    games_howell_interaction = pg.pairwise_gameshowell(data=df, dv='Value', between='Interaction')
    
    ## Same as above
    # df['Group'] = df['Genotype'] + '_' + df['Treatment']
    # games_howell_results = pg.pairwise_gameshowell(dv='Value', between='Group', data=df)
    
    print("\nGames-Howell Post-Hoc Test Results for Interaction:")
    print(games_howell_interaction)
else:
    print("\nNo significant interaction effect found. Skipping Games-Howell post-hoc test for interaction.")

# Perform Games-Howell post-hoc test for main effects if they are significant
if p_genotype < 0.05:
    print("\nPerforming Games-Howell post-hoc test for Genotype...")
    games_howell_genotype = pg.pairwise_gameshowell(data=df, dv='Value', between='Genotype')
    print("\nGames-Howell Post-Hoc Test Results for Genotype:")
    print(games_howell_genotype)
else:
    print("\nNo significant main effect of Genotype found.")

if p_treatment < 0.05:
    print("\nPerforming Games-Howell post-hoc test for Treatment...")
    games_howell_treatment = pg.pairwise_gameshowell(data=df, dv='Value', between='Treatment')
    print("\nGames-Howell Post-Hoc Test Results for Treatment:")
    print(games_howell_treatment)
else:
    print("\nNo significant main effect of Treatment found.")

# Combine post-hoc results into a summary table
summary_table = pd.DataFrame()

if p_interaction < 0.05:
    if p_genotype < 0.05 and p_treatment < 0.05: 
        summary_table = pd.concat([games_howell_genotype, games_howell_treatment, games_howell_interaction])
        summary_table.insert(0, 'Source', ['games_howell_genotype'] * len(games_howell_genotype) + 
                                     ['games_howell_treatment'] * len(games_howell_treatment) + 
                                     ['games_howell_interaction'] * len(games_howell_interaction))
    
    elif p_genotype >= 0.05 and p_treatment < 0.05:  
        summary_table = pd.concat([games_howell_treatment, games_howell_interaction])
        summary_table.insert(0, 'Source', ['games_howell_treatment'] * len(games_howell_treatment) + 
                                     ['games_howell_interaction'] * len(games_howell_interaction))
    
    elif p_genotype < 0.05 and p_treatment >= 0.05:      
        summary_table = pd.concat([games_howell_genotype, games_howell_interaction])
        summary_table.insert(0, 'Source', ['games_howell_genotype'] * len(games_howell_genotype) + 
                                     ['games_howell_interaction'] * len(games_howell_interaction))
        
    else: 
        print(' nothing is significant!')
        
elif p_interaction >= 0.05:
    if p_genotype < 0.05 and p_treatment < 0.05: 
        summary_table = pd.concat([games_howell_genotype, games_howell_treatment])
        summary_table.insert(0, 'Source', ['games_howell_genotype'] * len(games_howell_genotype) + 
                                     ['games_howell_treatment'] * len(games_howell_treatment))
   
    elif p_genotype >= 0.05 and p_treatment < 0.05:  
        summary_table = pd.concat([games_howell_treatment])
        summary_table.insert(0, 'Source', ['games_howell_treatment'] * len(games_howell_treatment))
    
    elif p_genotype < 0.05 and p_treatment >= 0.05:      
        summary_table = pd.concat([games_howell_genotype])
        summary_table.insert(0, 'Source', ['games_howell_genotype'] * len(games_howell_genotype))
    
    else: 
        print(' nothing is significant!')        
        
print('\n\n\n')
print(summary_table)
