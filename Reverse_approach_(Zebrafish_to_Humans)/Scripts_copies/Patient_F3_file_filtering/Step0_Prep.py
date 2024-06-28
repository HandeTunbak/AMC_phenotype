# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:33:33 2023

@author: Dr Hande Tunbak
"""

## Import libaries
import numpy as np
import pandas as pd
from zipfile import ZipFile
import os
import glob
import shutil
import re
import sys
import PySimpleGUI as sg


#%%
## Set base folder
base_path= 'C:/Users/Hande/OneDrive - Oxford Brookes University/Documents/Projects/Reverse_approach_(Zebrafish_to_Humans)/Patient_F3_file_filtering'

## Read filtering gene list
gene_filter= pd.read_csv('C:/Users/Hande/OneDrive - Oxford Brookes University/Documents/Projects/Reverse_approach_(Zebrafish_to_Humans)/Anophthalmia_Synonyms/Human_Anophthalmia_Synonyms_SingleColumn.csv')

## Read patient F3 file, with header in the top row
F3_file= pd.read_csv('C:/Users/Hande/OneDrive - Oxford Brookes University/Documents/Projects/Reverse_approach_(Zebrafish_to_Humans)/Patient_F3_file/F3_1%_all-probands_20221013b_csv.csv',
                     encoding='latin1', header=0)

## Create copy of patient F3 file
F3_file_corrected= F3_file.copy()

## Make corrections to copy of patient F3 file 
## Correction in 'gene' column 
F3_file_corrected['gene']= F3_file_corrected['gene'].replace('MARC2_', 'MARC2')
F3_file_corrected['gene']= F3_file_corrected['gene'].replace('Mar-02', 'MARC2')

## Correction in 'Gene.refGene2019' column 
F3_file_corrected['Gene.refGene2019']= F3_file_corrected['Gene.refGene2019'].replace('Mar-02', 'MARC2')
F3_file_corrected['Gene.refGene2019']= F3_file_corrected['Gene.refGene2019'].replace('37316', 'MARC2')

## Correction in 'Gene.knownGene' column 
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('37315', 'MARCHF6')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('37316', 'MARC2')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('Mar-02', 'MARC2')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('37500', 'SEPTIN2')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('37680', 'MARCHF7')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('38046', 'MARCHF8')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('38776', 'MARCHF10')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('38230', 'SEPTIN8')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('38230', 'SEPTIN8')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('38777', 'MARCHF6')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('38960', 'SEPTIN10')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('39142', 'MARCHF7')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('39508', 'MARCHF8')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('39692', 'SEPTIN8')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('40057', 'SEPTIN9')
F3_file_corrected['Gene.knownGene']= F3_file_corrected['Gene.knownGene'].replace('40238', 'MARCHF10')

## Correction in 'Gene.symbol' column 
F3_file_corrected['Gene.symbol']= F3_file_corrected['Gene.symbol'].replace('37316', 'MARC2')
F3_file_corrected['Gene.symbol']= F3_file_corrected['Gene.symbol'].replace('Mar-02', 'MARC2')

## Collect columns where names are found in F3 file. Used to check the above correction are completed
Gene_name_cols_inF3= F3_file_corrected[['gene', 'Gene.refGene2019', 'Gene.knownGene', 'Gene.symbol' ]].copy()

## Search F3 file columns with gene names (Gene_name_cols_inF3)
## Create emtpy dataframe to store filtered F3 file (corrected F3)
filtered_data= pd.DataFrame()

## Search the corrected F4 file for 

        
for index, gene in enumerate(gene_filter['Gene_Names']): 
   print (gene)
   temp_filt_data= F3_file_corrected.loc[F3_file_corrected['gene'].isin([gene])]

   filtered_data = pd.concat([filtered_data, temp_filt_data])
            
## Save 
filtered_data.to_csv((base_path + '/Filtered_F3_data.csv'), index=False)


filtered_data= filtered_data.reset_index()
filtered_data['index']= filtered_data['index']-1
filtered_data.rename(columns={"index": "Excel index (position)"}, inplace=True)

#%%

family_nos= pd.DataFrame()

for family_no in F3_file_corrected['sample']:
    
    if family_no[1].isspace() or family_no[1].isalpha():
        
        temp1= family_no[0:1]
        temp1= pd.Series(temp1)
        family_nos= pd.concat([family_nos, temp1])    
    
    elif family_no[2].isspace() or family_no[2].isalpha():
        
        temp2= family_no[0:2]
        temp2= pd.Series(temp2)
        family_nos= pd.concat([family_nos, temp2])

    elif family_no[3].isspace() or family_no[3].isalpha():
        temp3= family_no[0:3]
        temp3= pd.Series(temp3)
        family_nos= pd.concat([family_nos, temp3])        
    
    else: 
        print(family_no)

family_nos.columns= ['Family No.']
family_nos= family_nos.reset_index(drop=True)

updated_filtered_data = F3_file_corrected
updated_filtered_data.insert(loc=0, column='Family No.', value=family_nos )

No_unique_families= updated_filtered_data['Family No.'].nunique()

## Save 
updated_filtered_data.to_csv((base_path + '/Corrected_F3_with_unique_family_no.csv'), index=False)


#%% Return the number of unique families

textfile = open((base_path + "/No_unique_families_in_cohort.txt"), mode="w")
textfile.write("Number of unique in cohort = " + str(No_unique_families))
textfile.close()
























