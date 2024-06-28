# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:43:08 2023

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
import openpyxl
from openpyxl.styles import PatternFill

#%%
## Set base folder
base_path= 'C:/Users/Hande/OneDrive - Oxford Brookes University/Documents/Projects/Reverse_approach_(Zebrafish_to_Humans)/Patient_F3_file_filtering'

amc= 'Anophthalmia'
#amc= 'Microphthalmia'
#amc= 'Coloboma'

#%% Organise input and output folders

if amc== 'Anophthalmia' or amc== 'anophthalmia' :
    input_folder= base_path + '/Step2_Patient_F3_Filtering_Return_All_Variants/Anophthalmia/Input'
   
    shutil.copy(base_path + '/Step1_Patient_F3_Filtering/Anophthalmia/Output/Final_Filtered_F3_data.csv'
               , (input_folder + '/Final_Filtered_F3_data.csv'))
               
    shutil.copy(base_path  + '/Step0_Correcting_F3_file/Output/Corrected_F3_nan.csv'
               , (input_folder + '/Corrected_F3_nan.csv'))  
        
    shutil.copy(base_path  + '/Step1_Patient_F3_Filtering/Anophthalmia/Input/Human_Anophthalmia_Synonyms_SingleColumn.csv'
               , (input_folder + '/Human_Anophthalmia_Synonyms_SingleColumn.csv'))  
                   
    input_file= input_folder + '/Final_Filtered_F3_data.csv'
    
    output_folder= base_path  + '/Step2_Patient_F3_Filtering_Return_All_Variants/Anophthalmia/Output'

    genes= input_folder + '/Human_Anophthalmia_Synonyms_SingleColumn.csv'
    

elif amc== 'Microphthalmia' or amc== 'microphthalmia' :   
    input_folder= base_path  + '/Step2_Patient_F3_Filtering_Return_All_Variants/Microphthalmia/Input'
    
    shutil.copy(base_path + '/Step1_Patient_F3_Filtering/Microphthalmia/Output/Final_Filtered_F3_data.csv'
                , (input_folder + '/Final_Filtered_F3_data.csv'))
               
    shutil.copy(base_path  + '/Step0_Correcting_F3_file/Output/Corrected_F3_nan.csv'
               , (input_folder + '/Corrected_F3_nan.csv'))  

    shutil.copy(base_path  + '/Step1_Patient_F3_Filtering/Microphthalmia/Input/Human_Microphthalmia_Synonyms_SingleColumn.csv'
               , (input_folder + '/Human_Microphthalmia_Synonyms_SingleColumn.csv'))  
        
    input_file= input_folder + '/Final_Filtered_F3_data.csv'
   
    output_folder= base_path  + '/Step2_Patient_F3_Filtering_Return_All_Variants/Microphthalmia/Output'

    genes= input_folder + '/Human_Microphthalmia_Synonyms_SingleColumn.csv'
    

elif amc== 'Coloboma' or amc== 'coloboma' :  
    input_folder= base_path  + '/Step2_Patient_F3_Filtering_Return_All_Variants/Coloboma/Input'
    
    shutil.copy(base_path + '/Step1_Patient_F3_Filtering/Coloboma/Output/Final_Filtered_F3_data.csv'
               , (input_folder + '/Final_Filtered_F3_data.csv'))
               
    shutil.copy(base_path  + '/Step0_Correcting_F3_file/Output/Corrected_F3_nan.csv'
               , (input_folder + '/Corrected_F3_nan.csv'))  
    
    shutil.copy(base_path  + '/Step1_Patient_F3_Filtering/Coloboma/Input/Human_Coloboma_Synonyms_SingleColumn.csv'
               , (input_folder + '/Human_Coloboma_Synonyms_SingleColumn.csv'))  
     
    input_file= input_folder + '/Final_Filtered_F3_data.csv'
    
    output_folder= base_path  + '/Step2_Patient_F3_Filtering_Return_All_Variants/Coloboma/Output'

    genes= input_folder + '/Human_Coloboma_Synonyms_SingleColumn.csv'
    
else:
        print('You have not selected which AMC list you want to use')
    

#%% Read file and create mask
 
## Read patient F3 file, with header in the top row
F3_file= pd.read_csv(input_folder + '/Corrected_F3_nan.csv', encoding='latin1', header=0, 
                     dtype={'wgRNA': str,
                            'MutPred_score': float,
                            'GTEx_V6p_gene': str, 
                            'GTEx_V6p_tissue': str,
                            'TIGRpanel_refseq': str,
                            'AMC_genes_March2020': str }  )   
 
## Read filtered F3 file which will be used to obtain the Family numbers that appear in the Anophthalmia/Microphthalmia/ Coloboma list      
Family_nos= pd.read_csv(input_folder + '/Final_Filtered_F3_data.csv', encoding='latin1', header=0)   

## Read gene list including synonmys               
genes_filter= pd.read_csv(genes)

## Create mask for filtering. This matches where Family numbers appear in the F3 file         
mask = F3_file.Family_No.isin(Family_nos.Family_No)

## Apply mask to F3 file
masked_F3_file = F3_file[mask]  
        
## Sort masked F3 file according to the variant in the gene and the family number
masked_F3_file = masked_F3_file.sort_values(['Family_No','gene'], ascending =[True, True])


#%%
## Save 

## As .csv file 
masked_F3_file.to_csv((output_folder +  '/' + amc + '_geneList_All_variants.csv'), index=False)  

## As excel file
masked_F3_file.to_excel((output_folder +  '/' + amc + '_geneList_All_variants.xlsx'), index=False)    

#%% Return the number of unique families

No_unique_families= masked_F3_file['Family_No'].nunique()

## Store number of familyies in text file
textfile = open((output_folder  + "/No_unique_filtered_families.txt"), mode="w")
textfile.write("Number of unique families following filtering = " + str(No_unique_families))
textfile.close() 

#%% Colourise masked file

df= masked_F3_file.copy()
#df= pd.read_excel((output_folder +  '/Anophthalmia_geneList_All_variants.xlsx'), sheet_name='Sheet1')                                      
                                   
def color(x):
    c1 = 'color: gray' #gray 
    c2 = 'color: gray' #gray 
    c3 = 'color: gray' #gray 
    c4 = 'font-weight: bold ; background-color: #D3D3D3'
    c5 = 'color: gray' #set font colour to grey 
    c6 = 'color: gray' #set font colour to grey 
    c = '' 
    
    #compare columns
    mask1 = (x['CLNSIG'].str.contains('benign|Benign').fillna(False))
    #mask1 = (x['CLNSIG'] == 'Benign') 
    mask2 = (x['InterVar_automated'].str.contains('benign|Benign').fillna(False)) 
    #mask2 = (x['InterVar_automated'] == 'Likely_benign') 
    mask3 = ((x['CADD_phred']).astype(float) < 15)
    mask4 = (x['gene'].isin(genes_filter['Gene_Names']))   
    mask5 = ((x['Constructated_min_allel_freqs']).astype(float) >= 0.01 ) & (x.Constructated_min_allel_freqs.notnull() )
    mask6 = (x['Func.refGene'] != ('exonic' or 'exonic;splicing' or 'ncRNA_exonic' or 'ncRNA_exonic;splicing' or 'splicing'))    
            
    #DataFrame with same index and columns names as original filled empty strings
    df1 =  pd.DataFrame(c, index=x.index, columns=x.columns)
    #modify values of df1 column by boolean mask
    #df1.loc[mask1, :] = c1
    df1.loc[mask2, :] = c2
    df1.loc[mask3, :] = c3
    df1.loc[mask4] = c4     #df1.loc[mask4, :] = c4 #is returning the same as df1.loc[mask4] = c4 ?
    df1.loc[mask5, :] = c5
    df1.loc[mask6, :] = c6
    
    return df1


#df.style.apply(color, axis=None)

## Apply style to dataframe using the definition above
dt = df.style.apply(color, axis=None)  

## Save styled dataframe as .xlsx file for opnenning with excel.
if amc== 'Anophthalmia' or amc== 'anophthalmia' :
    dt.to_excel((output_folder +  '/Anophthalmia_geneList_All_variants_greyout.xlsx'), index=False)  
    
elif amc== 'Microphthalmia' or amc== 'microphthalmia' :  
    dt.to_excel((output_folder +  '/Microphthalmia_geneList_All_variants_greyout.xlsx'), index=False)
    
elif amc== 'Coloboma' or amc== 'coloboma' : 
    dt.to_excel((output_folder +  '/Coloboma_geneList_All_variants_greyout.xlsx'), index=False)
    
else:
        print('You have not selected which AMC list you want to use')



#%% Unused code:

## Not used 
# ## Replace instances of '.' in CADD_phred column with nan - if not already done in step 0
# masked_F3_file['CADD_phred'] = masked_F3_file['CADD_phred'].replace(to_replace=['.'], value= np.nan)  

# ## Save 
# ## As .csv file 
# masked_F3_file.to_csv((output_folder +  '/Anophthalmia_geneList_All_variants_CADDPhred_nan.csv'), index=False)         
        
# ## As excel file
# masked_F3_file.to_excel((output_folder +  '/Anophthalmia_geneList_All_variants_CADDPhred_nan.xlsx'), index=False)           
        
##--------------------------------------------------      

## Not used
# masked_F3_file['CADD_phred'] = masked_F3_file['CADD_phred'].replace(to_replace=['.'], value= np.nan) 
# masked_F3_file['Constructated_min_allel_freqs'] = masked_F3_file['Constructated_min_allel_freqs'].replace(to_replace=['.'], value= np.nan) 

##--------------------------------------------------      
 

# def color(x):
#     c1 = 'color: gray' #gray 
#     c2 = 'color: #D3D3D3' #lightgray / lightgrey same as gray
#     c3 = 'color: #D3D3D3' #lightgray / lightgrey
#     c4 = ('font-weight: bold' and 'background-color: #D3D3D3')
#     c = '' 
    
#     #compare columns
#     mask1 = (x['CLNSIG'] == 'Benign') 
#     mask2 = (x['InterVar_automated'] == 'Likely_benign') 
#     mask3 = ((x['CADD_phred']).astype(float) < 15)
#     mask4 = (x['gene'].isin(genes_filter['Gene_Names']))       
            
#     #DataFrame with same index and columns names as original filled empty strings
#     df1 =  pd.DataFrame(c, index=x.index, columns=x.columns)
#     #modify values of df1 column by boolean mask
#     df1.loc[mask1, :] = c1
#     df1.loc[mask2, :] = c2
#     df1.loc[mask3, :] = c3
#     df1.loc[mask4, :] = c4 #is returning teh same as df1.loc[mask4] = c4
    
    
#     return df1


# #df.style.apply(color, axis=None)

# dt = df.style.apply(color, axis=None)  

# dt.to_excel((output_folder +  '/Anophthalmia_geneList_All_variants_greyoutnanb2.xlsx'), index=False)  


# def highlight_rows1(row):
#     col1 = row.loc['CLNSIG']
#     col2 = row.loc['InterVar_automated']
#     if col1 == 'Benign':
#         color = '#D3D3D3' #lightgray / lightgrey
#     elif col2 == 'Likely_benign':
#         color = '#D3D3D3' #lightgray / lightgrey  
#         return ['color: {}'.format(color) for r in row] # change color to background-color to select background color



# def highlight_rows2(row):
#     col2 = row.loc['InterVar_automated']
#     if col2 == 'Likely_benign':
#         color = '#D3D3D3' #lightgray / lightgrey
#         return ['color: {}'.format(color) for r in row] # change color to background-color to select background color


# # def highlight_rows2(row):
# #     #col3 = row.loc['CADD_phred']
    
# #     if col1 == 'Benign' or col2 == 'Likely_benign':
# #         color = '#D3D3D3' #lightgray / lightgrey
# #         return ['color: {}'.format(color) for r in row] # change color to background-color to select background color
    
    
# dt = df.style.apply(highlight_rows1, axis=1)   
# dt2=  dt.style.apply(highlight_rows2, axis=1)    


# dt.to_excel((output_folder +  '/Anophthalmia_geneList_All_variants_greyout.xlsx'), index=False)  






























































# masked_F3_file2 = masked_F3_file
# def _color_red_or_green(val):
#     color = 'red' if val < 10 else 'green'
#     return 'color: %s' % color


# x= masked_F3_file2.style.applymap(_color_red_or_green, subset=['Family_No'])

# x.to_excel((output_folder +  '/Anophthalmia_geneList_All_variants_reds.xlsx'), index=False)  


# masked_F3_file2 = masked_F3_file

# def highlight_col(x):
#     #copy df to new - original data are not changed
#     df = x.copy()
#     #set by condition
#     mask = df['Family_No'] < 10
#     df.loc[mask, :] = ':green'
#     df.loc[~mask,:] = ':black'
#     return df    

# z= masked_F3_file2.style.apply(highlight_col, axis=None)

# z.to_excel((output_folder +  '/Anophthalmia_geneList_All_variants_green.xlsx'), index=False) 











        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        