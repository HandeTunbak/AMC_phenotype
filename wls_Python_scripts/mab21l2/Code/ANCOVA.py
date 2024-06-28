# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Dr Hande Tunbak)
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


input_file_path = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep/'

x= 2

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

outcome = pd.concat([file['Injected_MAB_siblings'], file['Injected_MAB_phenos']])
baesline= pd.concat([file['Control_MAB_siblings'], file['Control_MAB_phenos']])
genotype_a= pd.DataFrame(df = pd.DataFrame([['control'] * 1] * len(file[''])))

# Load your dataset
# Assuming your dataset is named 'data' and contains columns: 'outcome', 'genotype', 'treatment', and 'baseline'
# Replace 'data.csv' with the actual file path if loading from a CSV file
data = pd.read_csv('data.csv')

# Fit the ANCOVA model
# 'outcome' is the dependent variable, 'genotype' and 'treatment' are the categorical predictors (factors),
# and 'baseline' is the covariate
model = ols('outcome ~ genotype + treatment + baseline + genotype:treatment', data=data).fit()

# Print the ANCOVA summary
print(model.summary())


Make sure to replace 'data.csv' with the actual file path if loading your data from a CSV file. Also, ensure that your dataset includes columns for the outcome variable ('outcome'), categorical predictors ('genotype' and 'treatment'), and the covariate ('baseline').

This code fits an ANCOVA model using ordinary least squares (OLS) regression with interaction between 'genotype' and 'treatment'. The model.summary() command prints a summary of the ANCOVA results, including coefficients, standard errors, p-values, and other relevant statistics.

Before running this code, ensure that you have the pandas and statsmodels libraries installed in your Python environment. You can install them via pip if you haven't already:


pip install pandas statsmodels
