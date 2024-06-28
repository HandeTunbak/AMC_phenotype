# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:59:13 2023

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
F3_file= pd.read_csv(base_path + '/Corrected_F3_with_unique_family_no.csv', encoding='latin1', header=0)

## Create copy of patient F3 file
F3_file_corrected= F3_file.copy()

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

# ## Set rows to keep in full patient F3 file
# rows_to_keep= (pd.DataFrame(masked_F3.index))
# rows_to_keep.columns=['masked_F3_indexes']

# ## Filter patient F3 file with rows to keep
# filtered_F3= pd.DataFrame()
# for row in rows_to_keep['masked_F3_indexes']:
#     nextrow= row +1
#     temp= F3_file_corrected[row:nextrow]
#     filtered_F3= pd.concat([filtered_F3, temp])
#     # print(row)
#     # print(nextrow)

filtered_data= filtered_data.reset_index()
filtered_data['index']= filtered_data['index']-1
filtered_data.rename(columns={"index": "Excel index (position)"}, inplace=True)


#%%

# updated_filtered_data= pd.concat([filtered_data, family_nos], axis=1, ignore_index=False)

updated_filtered_data = filtered_data

No_unique_families= updated_filtered_data['Family No.'].nunique()

## Save 
updated_filtered_data.to_csv((base_path + '/Final_Filtered_F3_data.csv'), index=False)

#%% Return the number of unique families

textfile = open((base_path + "/No_unique_filtered_families.txt"), mode="w")
textfile.write("Number of unique famiiles following filtering = " + str(No_unique_families))
textfile.close()
























