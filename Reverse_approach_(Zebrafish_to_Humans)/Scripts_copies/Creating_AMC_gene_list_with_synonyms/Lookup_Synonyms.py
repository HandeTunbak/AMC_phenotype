# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Dr Hande Tunbak)
"""

import pandas as pd
import numpy as np


#%%
base_folder= 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/Reverse_approach_(Zebrafish_to_Humans)'

file_path= 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/Reverse_approach_(Zebrafish_to_Humans)/Summary of filtered genes -panelAPP genes removed_v2.xlsx'

Human_gene_synonyms_path = 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/Reverse_approach_(Zebrafish_to_Humans)/Human_gene_synonyms_corrected_v2.csv'
#%%

## Open csv file
Anophthalmia= pd.read_excel(file_path, sheet_name='Summary_Anophthalmia', header=(0))

Coloboma= pd.read_excel(file_path, sheet_name='Summary_Coloboma', header=(0))

Microphthalmia= pd.read_excel(file_path, sheet_name='Summary_Microphthalmia', header=(0))

Human_gene_synonyms_list= pd.read_csv(Human_gene_synonyms_path, header=(0), on_bad_lines='skip')


#%% Remove wrong/ old gene synonyms to prevent confusion

#   Replace any emtpy strongs in the columns with np.nan objects
Human_gene_synonyms_list['Gene name'].replace('', np.nan, inplace=True)

#   Drop the null values:
Human_gene_synonyms_list.dropna(subset=['Gene name'], inplace=True)   

Human_gene_synonyms_list = Human_gene_synonyms_list


#%%
Anophthalmia_synonyms_list= pd.DataFrame()

for gene in (Anophthalmia['Human']):
    print(gene)
    Anophthalmia_synonyms= Human_gene_synonyms_list.loc[(Human_gene_synonyms_list['Gene name'] == gene )]
    Anophthalmia_synonyms_list= pd.concat([Anophthalmia_synonyms_list, Anophthalmia_synonyms], ignore_index=True)

## Save   
Anophthalmia_synonyms_list.to_csv((base_folder + '/Anophthalmia_Synonyms' + '/Human_Anophthalmia_Synonyms_MultipleColumns.csv'), index=False)


Unique_Anophthalmia_synonyms_list= []  

Anophthalmia_synonyms_list = Anophthalmia_synonyms_list.drop(['Gene stable ID'], axis= 1)

for column in (Anophthalmia_synonyms_list):
    print(column)
    index= 0
    
    for row_numbers in range(len(Anophthalmia_synonyms_list)):
        print(row_numbers)
    
        if pd.isna(Anophthalmia_synonyms_list[column][row_numbers]):
            pass
        else: 
            Unique_Anophthalmia_synonyms_list.append(Anophthalmia_synonyms_list[column][row_numbers])
    index= index+1
    
Unique_Anophthalmia_synonyms_list= pd.DataFrame(Unique_Anophthalmia_synonyms_list ) 
Unique_Anophthalmia_synonyms_list.columns= ['Gene_Names']  
Unique_Anophthalmia_synonyms_list= Unique_Anophthalmia_synonyms_list['Gene_Names'].drop_duplicates()


## Save
Unique_Anophthalmia_synonyms_list.to_csv((base_folder + '/Anophthalmia_Synonyms' + '/Human_Anophthalmia_Synonyms_SingleColumn.csv'), index=False)



#%%

Coloboma_synonyms_list= pd.DataFrame()

for gene in (Coloboma['Human']):
    print(gene)
    Coloboma_synonyms= Human_gene_synonyms_list.loc[(Human_gene_synonyms_list['Gene name'] == gene )]
    Coloboma_synonyms_list= pd.concat([Coloboma_synonyms_list, Coloboma_synonyms], ignore_index=True)

## Save   
Coloboma_synonyms_list.to_csv((base_folder + '/Coloboma_Synonyms' + '/Human_Coloboma_Synonyms_MultipleColumns.csv'), index=False)


Unique_Coloboma_synonyms_list= []  

Coloboma_synonyms_list = Coloboma_synonyms_list.drop(['Gene stable ID'], axis= 1)

for column in (Coloboma_synonyms_list):
    print(column)
    index= 0
    
    for row_numbers in range(len(Coloboma_synonyms_list)):
        print(row_numbers)
    
        if pd.isna(Coloboma_synonyms_list[column][row_numbers]):
            pass
        else: 
            Unique_Coloboma_synonyms_list.append(Coloboma_synonyms_list[column][row_numbers])
    index= index+1
    
Unique_Coloboma_synonyms_list= pd.DataFrame(Unique_Coloboma_synonyms_list ) 
Unique_Coloboma_synonyms_list.columns= ['Gene_Names']  
Unique_Coloboma_synonyms_list= Unique_Coloboma_synonyms_list['Gene_Names'].drop_duplicates()

## Save
Unique_Coloboma_synonyms_list.to_csv((base_folder + '/Coloboma_Synonyms' +'/Human_Coloboma_Synonyms_SingleColumn.csv'), index=False)


#%%

Microphthalmia_synonyms_list= pd.DataFrame()

for gene in (Microphthalmia['Human']):
    print(gene)
    Microphthalmia_synonyms= Human_gene_synonyms_list.loc[(Human_gene_synonyms_list['Gene name'] == gene )]
    Microphthalmia_synonyms_list= pd.concat([Microphthalmia_synonyms_list, Microphthalmia_synonyms], ignore_index=True)

## Save   
Microphthalmia_synonyms_list.to_csv((base_folder + '/Microphthalmia_Synonyms' + '/Human_Microphthalmia_Synonyms_MultipleColumns.csv'), index=False)


Unique_Microphthalmia_synonyms_list= []  

Microphthalmia_synonyms_list = Microphthalmia_synonyms_list.drop(['Gene stable ID'], axis= 1)

for column in (Microphthalmia_synonyms_list):
    print(column)
    index= 0
    
    for row_numbers in range(len(Microphthalmia_synonyms_list)):
        print(row_numbers)
    
        if pd.isna(Microphthalmia_synonyms_list[column][row_numbers]):
            pass
        else: 
            Unique_Microphthalmia_synonyms_list.append(Microphthalmia_synonyms_list[column][row_numbers])
    index= index+1
    
Unique_Microphthalmia_synonyms_list= pd.DataFrame(Unique_Microphthalmia_synonyms_list ) 
Unique_Microphthalmia_synonyms_list.columns= ['Gene_Names']  
Unique_Microphthalmia_synonyms_list= Unique_Microphthalmia_synonyms_list['Gene_Names'].drop_duplicates()

## Save
Unique_Microphthalmia_synonyms_list.to_csv((base_folder + '/Microphthalmia_Synonyms' + '/Human_Microphthalmia_Synonyms_SingleColumn.csv'), index=False)

#%%







