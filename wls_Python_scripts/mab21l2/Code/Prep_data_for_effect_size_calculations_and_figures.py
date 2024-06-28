# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Dr Hande Tunbak)
"""

'''If fisrt time using code, install dabest via pip:
    pip install dabest'''
    
import glob
import numpy as np
import scipy
import pandas as pd
import math
import dabest
from decimal import Decimal

#%% # Set paths for csv files  
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Location with subdirectories
Experiment_folder = 'C:/Users/Hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_MAB/Testing_area/Effect_size_prep'

csv_file_paths= glob.glob(Experiment_folder + '/' + '**.csv', recursive=False )
    
for file in csv_file_paths:
    
    file_name= file.split('Effect_size_prep')
    file_name= file_name[1][1:]
    
    # # Load csv file 
    csv_data= pd.read_csv( file )     
    
    control_MAB_siblings = pd.DataFrame(index=range(csv_data['Control_MAB_siblings'].count()),columns=range(4))
    control_MAB_siblings.columns = ['Condition', 'Genotype', 'Treatment', 'Value']
    control_MAB_siblings['Condition'] = 'Control'
    control_MAB_siblings['Genotype'] = 'mab_sibs'
    control_MAB_siblings['Treatment'] = ' '
    control_MAB_siblings['Value'] = csv_data['Control_MAB_siblings']
    control_MAB_siblings = control_MAB_siblings.reset_index(drop=True)
    
    control_MAB_phenos = pd.DataFrame(index=range(csv_data['Control_MAB_phenos'].count()),columns=range(4))
    control_MAB_phenos.columns = ['Condition', 'Genotype', 'Treatment', 'Value']
    control_MAB_phenos['Condition'] = 'Control'
    control_MAB_phenos['Genotype'] = 'mab-/-'
    control_MAB_phenos['Treatment'] = ' '
    control_MAB_phenos['Value'] = csv_data['Control_MAB_phenos']
    control_MAB_phenos = control_MAB_phenos.reset_index(drop=True)
    
    injected_MAB_siblings = pd.DataFrame(index=range(csv_data['Injected_MAB_siblings'].count()),columns=range(4))
    injected_MAB_siblings.columns = ['Condition', 'Genotype', 'Treatment', 'Value']
    injected_MAB_siblings['Condition'] = 'Test'
    injected_MAB_siblings['Genotype'] = 'mab_sibs'
    injected_MAB_siblings['Treatment'] = 'wls-/-'
    injected_MAB_siblings['Value'] = csv_data['Injected_MAB_siblings']
    injected_MAB_siblings = injected_MAB_siblings.reset_index(drop=True)
    
    injected_MAB_phenos = pd.DataFrame(index=range(csv_data['Injected_MAB_phenos'].count()),columns=range(4))
    injected_MAB_phenos.columns = ['Condition', 'Genotype', 'Treatment', 'Value']
    injected_MAB_phenos['Condition'] = 'Test'
    injected_MAB_phenos['Genotype'] = 'mab-/-'
    injected_MAB_phenos['Treatment'] = 'wls-/-'
    injected_MAB_phenos['Value'] = csv_data['Injected_MAB_phenos']
    injected_MAB_phenos= injected_MAB_phenos.reset_index(drop=True)
    
    # Creat dataframe to store the data
    df_list= [control_MAB_siblings, control_MAB_phenos, injected_MAB_siblings, injected_MAB_phenos]
    data = pd.concat(df_list, axis=0, ignore_index=True)
    #data= data.reset_index()


    # Save dataframe 
    data.to_csv( Experiment_folder + '/Inputs_csv_files' + str('/' + 'Input_' + file_name), index=False)

    unpaired_delta2 = dabest.load(data = data, x = ["Treatment", "Condition"], y = "Value", delta2 = True, experiment = "Genotype")
    
    # Set variables 
    variable1= unpaired_delta2.mean_diff.statistical_tests['difference'][0]
    variable2= unpaired_delta2.mean_diff.statistical_tests['bca_low'][0]
    variable3= unpaired_delta2.mean_diff.statistical_tests['bca_high'][0]
    variable4= '{:#.3g}'.format(unpaired_delta2.mean_diff.results['pvalue_permutation'][0])
   
    #  # if you want it as signficant figures use->> variable4= '%.2E' % Decimal(unpaired_delta2.mean_diff.results['pvalue_permutation'][0])
        
    variable5= unpaired_delta2.mean_diff.statistical_tests['difference'][1]
    variable6= unpaired_delta2.mean_diff.statistical_tests['bca_low'][1]
    variable7= unpaired_delta2.mean_diff.statistical_tests['bca_high'][1]
    variable8= '{:#.3g}'.format(unpaired_delta2.mean_diff.results['pvalue_permutation'][0])
    
    variable9=  unpaired_delta2.mean_diff.delta_delta.difference
    variable10= unpaired_delta2.mean_diff.delta_delta.bca_low
    variable11= unpaired_delta2.mean_diff.delta_delta.bca_high
    variable12= '{:#.3g}'.format(unpaired_delta2.mean_diff.delta_delta.pvalue_permutation)

    
    figures_folder= 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/WLS/Code_testing/WLS_Combined_MAB_and_TL/Testing_area/Figures_and_Stats/Effect_size_figures/'
   
    # Create report file
    reportFilename = ( figures_folder +  file_name[:-4] + '.txt')
    reportFile = open(reportFilename, 'w')

    line0 = ('Effect size(s) with 95% confidence intervals will be computed for:' + '\n' +
             '  1. M Placebo minus W Placebo' + '\n' +
             '   2. M Drug minus W Drug' + '\n' +
             '   3. Drug minus Placebo (only for mean difference)' + '\n' +
             '5000 resamples will be used to generate the effect size bootstraps.' + '\n' + '\n' )
    
             

    line1 = ('The unpaired mean difference between mab_sibs and wls-/- mab_sibs is: ' 
             + str(variable1) + ' [' + '95%CI ' + str(variable2) + ', ' + str(variable3) + '].' + '\n')
    
    line2 = str('The P value of the two-sided permutation t-test is ' + str(variable4) + '.' + '\n' + '\n')
    
    
    
    line3 = ('The unpaired mean difference between mab-/- and wls-/- mab-/- is: ' 
             + str(variable5) + ' [' + '95%CI ' + str(variable6) + ', ' + str(variable7) + '].' + '\n')
    
    line4 = str('The P value of the two-sided permutation t-test is ' + str(variable8) + '.' + '\n' + '\n')
    
    
    
    line5 = ('The delta-delta between mab_sibs and mab-/- is: ' 
             + str(variable9) + ' [' + '95%CI ' + str(variable10) + ', ' + str(variable11) + '].' + '\n')
    
    line6 = str('The P value of the two-sided permutation t-test is ' + str(variable12) + '.' + '\n' + '\n')
    

    reportFile.write(line0 +  
                     line1 +
                     line2 + 
                     line3 + 
                     line4 + 
                     line5 + 
                     line6  )

    reportFile.close()


#%% Useful websites

# https://www.estimationstats.com/#/analyze/delta-delta
# https://www.estimationstats.com/#/
# https://acclab.github.io/DABEST-python/blog/ 