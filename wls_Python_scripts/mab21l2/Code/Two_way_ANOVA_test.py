# -*- coding: utf-8 -*-
"""
Created on %(date)s

If using code fro first time install pingouin
        $ pip install pingouin

@author: %(Dr Hande Tunbak)
"""

import pandas as pd
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools
import pingouin as pg
import numpy as np

#%%

input_file_path = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep/Inputs_csv_files/'

x= 3

x=x-1

if x==1:
    input_file_name= 'Input_Average_Body_Length_(um).csv'
    
elif x==2: 
    input_file_name= 'Input_Eccentricity.csv'
    
elif x==3:  
    input_file_name= 'Input_Eye_Area_(um^2).csv'
    
elif x==4:  
    input_file_name= 'Input_Eye_Body_Ratio.csv'
    
elif x==5:   
    input_file_name= 'Input_Eye_Head_Ratio.csv'
    
elif x==6:     
    input_file_name= 'Input_Eye_Lengths_(um).csv'
    
elif x==7:   
    input_file_name= 'Input_Head_Body_Ratio.csv'
else: 
    print('error') 


input_file= str(input_file_path + input_file_name)

file= pd.read_csv(input_file)


# Performing two-way ANOVA 
model = ols('Value ~ Genotype+Treatment+Genotype:Treatment', data=file).fit() 
anova_table= sm.stats.anova_lm(model, typ=2) 

print(anova_table)

#%% Interaction plot

sns.set(style='whitegrid')

# Create interaction plot
fig, ax = plt.subplots(figsize=(8, 6))

# Use seaborn's interaction plot functionality
df= file
#sns.pointplot(data=df, x='Genotype', y='Value', hue='Treatment', markers=["o", "s"], linestyles=["-", "--"], ax=ax)

sns.pointplot(data=df, x='Treatment', y='Value', hue='Genotype', markers=["o", "s"], linestyles=["-", "--"], ax=ax)


# Customize the plot
ax.set_title('Interaction Plot of Genotype and Treatment')
ax.set_xlabel('Genotype')
ax.set_ylabel('Value')
plt.legend(title='Treatment')
plt.show()


#%% Post-hoc test

# Conduct post-hoc test using Tukey's HSD
# Combine factors for the post-hoc test

df2= df
df2= df2.drop('Condition', axis=1)

# Combine factors for the post-hoc test
df2['Group'] = df2['Genotype'] + ':' + df2['Treatment']


#%% Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=df2['Value'], groups=df2['Group'], alpha=0.05)

# Print results of Tukey's HSD test
print(tukey)


# Plot the results of Tukey's HSD test
tukey.plot_simultaneous()
plt.show()


#%%
# Perform pairwise t-tests
comparisons = list(itertools.combinations(df2['Group'].unique(), 2))
p_values = []

for group1, group2 in comparisons:
    data1 = df2[df2['Group'] == group1]['Value']
    data2 = df2[df2['Group'] == group2]['Value']
    t_stat, p_value, t_value = sm.stats.ttest_ind(data1, data2)
    p_values.append(p_value)

# Apply Bonferroni correction
_, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

# Print results
comparison_results = pd.DataFrame({
    'Comparison': comparisons,
    'p_value': p_values,
    'p_value_corrected': p_values_corrected
})

print("\nPairwise Comparisons with Bonferroni Correction")
print(comparison_results)

#%% Games-Howell

# Perform Two-Way ANOVA
anova = pg.anova(data=df2, dv='Value', between=['Genotype', 'Treatment'], detailed=True)


# Perform Two-Way ANOVA
anova_results = pg.anova(data=df, dv='Value', between=['Genotype', 'Treatment'], detailed=True)
# print("Two-Way ANOVA Results:")
# print(anova_results)

# Check for assumptions
# Shapiro-Wilk test for normality
shapiro_a = pg.normality(data=df, dv='Value', group='Genotype')
# print("\nShapiro-Wilk Normality Test for Genotype:")
# print(shapiro_a)

shapiro_b = pg.normality(data=df, dv='Value', group='Treatment')
print("\nShapiro-Wilk Normality Test for Treatment:")
# print(shapiro_b)

# Combine the factors to perform Levene's test for the interaction
df['Genotype_B'] = df['Genotype'] + '_' + df['Treatment']

# # Levene's test for variance homogeneity
levene_a = pg.homoscedasticity(data=df, dv='Value', group='Genotype')
print("\nLevene's Test for Variance Homogeneity for Factor A:")
# print(levene_a)

levene_b = pg.homoscedasticity(data=df, dv='Value', group='Treatment')
print("\nLevene's Test for Variance Homogeneity for Factor B:")
# print(levene_b)

levene_ab = pg.homoscedasticity(data=df, dv='Value', group='Genotype_B')
# print("\nLevene's Test for Variance Homogeneity for Interaction of Factors A and B:")
# print(levene_ab)

# Perform Games-Howell post-hoc test
games_howell_results = pg.pairwise_gameshowell(dv='Value', between='Genotype', data=df)
print("\nGames-Howell Test Results:")
print(games_howell_results)

#games_howell_interaction = pg.pairwise_gameshowell(data=df, dv='Value', between=['Genotype', 'Treatment'])
#%%

# Perform permutation-based ANOVA manually
def permutation_anova(df, dv, between, within, n_perm=1000):
    observed_f = pg.anova(data=df, dv=dv, between=between, detailed=True).round(6)['F'][0]
    permuted_fs = []
    for _ in range(n_perm):
        permuted_df = df.copy()
        permuted_df[within] = np.random.permutation(permuted_df[within])
        permuted_f = pg.anova(data=permuted_df, dv=dv, between=between, detailed=True).round(6)['F'][0]
        permuted_fs.append(permuted_f)
    p_value = (np.sum(permuted_fs >= observed_f) + 1) / (n_perm + 1)
    return observed_f, p_value

# Example usage
observed_f, p_value = permutation_anova(df, dv='Value', between='Genotype', within='Treatment', n_perm=5000)
print("Observed F-statistic:", observed_f)
print("p-value:", p_value)


#%%
### https://www.psychometrica.de/effect_size.html