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

amc= 'Anophthalmia'
amc= 'Microphthalmia'
#amc= 'Coloboma'

if amc== 'Anophthalmia' or amc== 'anophthalmia' :
    input_folder= base_path + '/Step1_Patient_F3_Filtering/Anophthalmia/Input'
   
    shutil.copy('C:/Users/Hande/OneDrive - Oxford Brookes University/Documents/Projects/Reverse_approach_(Zebrafish_to_Humans)/Lookup_Synonyms/Output/Anophthalmia_Synonyms/Human_Anophthalmia_Synonyms_SingleColumn.csv'
               , (input_folder + '/Human_Anophthalmia_Synonyms_SingleColumn.csv'))
               
    shutil.copy(base_path  + '/Step0_Correcting_F3_file/Output/Corrected_F3.csv'
               , (input_folder + '/Corrected_F3.csv'))  
               
    input_file= input_folder + '/Human_Anophthalmia_Synonyms_SingleColumn.csv'
    
    output_folder= base_path  + '/Step1_Patient_F3_Filtering/Anophthalmia/Output'

    
elif amc== 'Microphthalmia' or amc== 'microphthalmia' :   
    input_folder= base_path  + '/Step1_Patient_F3_Filtering/Microphthalmia/Input'
    
    shutil.copy('C:/Users/Hande/OneDrive - Oxford Brookes University/Documents/Projects/Reverse_approach_(Zebrafish_to_Humans)/Lookup_Synonyms/Output/Microphthalmia_Synonyms/Human_Microphthalmia_Synonyms_SingleColumn.csv'
                , (input_folder + '/Human_Microphthalmia_Synonyms_SingleColumn.csv'))
               
    shutil.copy(base_path  + '/Step0_Correcting_F3_file/Output/Corrected_F3.csv'
               , (input_folder + '/Corrected_F3.csv'))  
    
    input_file= input_folder + '/Human_Microphthalmia_Synonyms_SingleColumn.csv'
   
    output_folder= base_path  + '/Step1_Patient_F3_Filtering/Microphthalmia/Output'

    
elif amc== 'Coloboma' or amc== 'coloboma' :  
    input_folder= base_path  + '/Step1_Patient_F3_Filtering/Coloboma/Input'
    
    shutil.copy('C:/Users/Hande/OneDrive - Oxford Brookes University/Documents/Projects/Reverse_approach_(Zebrafish_to_Humans)/Lookup_Synonyms/Output/Coloboma_Synonyms/Human_Coloboma_Synonyms_SingleColumn.csv'
               , (input_folder + '/Human_Coloboma_Synonyms_SingleColumn.csv'))
               
    shutil.copy(base_path  + '/Step0_Correcting_F3_file/Output/Corrected_F3.csv'
               , (input_folder + '/Corrected_F3.csv'))  
    
    input_file= input_folder + '/Human_Coloboma_Synonyms_SingleColumn.csv'
    
    output_folder= base_path  + '/Step1_Patient_F3_Filtering/Coloboma/Output'

    
else:
        print('You have not selected which AMC list you want to use')
        
    
#%%    
## Read filtering gene list
gene_filter= pd.read_csv(input_file)

## Read patient F3 file, with header in the top row
F3_file= pd.read_csv(input_folder + '/Corrected_F3.csv', encoding='latin1', header=0, on_bad_lines='skip', dtype='unicode')

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
            

filtered_data= filtered_data.reset_index()
filtered_data['index']= filtered_data['index']+2
filtered_data.rename(columns={"index": "Excel index (position)"}, inplace=True)

## Save 
filtered_data.to_csv((output_folder +  '/Final_Filtered_F3_data.csv'), index=False)


#%% Return the number of unique families

# updated_filtered_data= pd.concat([filtered_data, family_nos], axis=1, ignore_index=False)

updated_filtered_data = filtered_data

No_unique_families= updated_filtered_data['Family_No'].nunique()

## Store number of familyies in text file
textfile = open((output_folder  + "/No_unique_filtered_families.txt"), mode="w")
textfile.write("Number of unique families following filtering = " + str(No_unique_families))
textfile.close()

#%% Colourise masked file

filtered_data['CADD_phred'] = filtered_data['CADD_phred'].replace(to_replace=['.'], value= np.nan) 
filtered_data['Constructated_min_allel_freqs'] = filtered_data['Constructated_min_allel_freqs'].replace(to_replace=['.'], value= np.nan) 
df= filtered_data.copy()
#df= pd.read_excel((output_folder +  '/Anophthalmia_geneList_All_variants.xlsx'), sheet_name='Sheet1')                                      
                                   
def color(x):
    c1 = 'color: gray' #set font colour to grey 
    c2 = 'color: gray' #set font colour to grey 
    c3 = 'color: gray' #set font colour to grey 
    c4 = 'font-weight: bold' #set font to bold
    c5 = 'color: gray' #set font colour to grey 
    c6 = 'color: gray' #set font colour to grey 
    c = '' 
    
    #compare columns
    mask1 = (x['CLNSIG'].str.contains('benign|Benign').fillna(False))
    #mask1 = (x['CLNSIG'] == 'Benign') 
    mask2 = (x['InterVar_automated'].str.contains('benign|Benign').fillna(False)) 
    #mask2 = (x['InterVar_automated'] == 'Likely_benign') 
    mask3 = ((x['CADD_phred']).astype(float) <= 15 ) & (x.CADD_phred.notnull() )# mask CADD Phred scores that are equal to or lower than 15. NB: CADD scores range from 1-99, where higher scores are more deleterious 
    mask4 = (x['gene'].isin(gene_filter['Gene_Names']))   
    mask5 = ((x['Constructated_min_allel_freqs']).astype(float) >= 0.01 ) & (x.Constructated_min_allel_freqs.notnull() )
    mask6 = (x['Func.refGene'] != ('exonic' or 'exonic;splicing' or 'ncRNA_exonic' or 'ncRNA_exonic;splicing' or 'splicing'))
            
    #DataFrame with same index and columns names as original filled empty strings
    df1 =  pd.DataFrame(c, index=x.index, columns=x.columns)
    #modify values of df1 column by boolean mask
    df1.loc[mask1, :] = c1
    df1.loc[mask2, :] = c2
    df1.loc[mask3, :] = c3
    df1.loc[mask5, :] = c5
    df1.loc[mask6, :] = c6

    return df1


#df.style.apply(color, axis=None)

dt = df.style.apply(color, axis=None)  

dt.to_excel((output_folder +  '/Final_Filtered_F3_data_greyout.xlsx'), index=False)  

























