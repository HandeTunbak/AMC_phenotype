# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:27:25 2023

@author: Hande
"""

"""
This file removes errors incorrect gene synonymns from the file downloaded from ensembl

"""

import pandas as pd
import numpy as np


#%%
base_folder= 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/Reverse_approach_(Zebrafish_to_Humans)'

file_path= 'C:/Users/Hande/OneDrive - Oxford Brookes University/Documents/Projects/Reverse_approach_(Zebrafish_to_Humans)/Files/Removing_incorrect_gene_synonyms/Input/Summary of filtered genes -panelAPP genes removed_v2.xlsx'

Human_gene_synonyms= 'C:/Users/Hande/OneDrive - Oxford Brookes University/Documents/Projects/Reverse_approach_(Zebrafish_to_Humans)/Files/Removing_incorrect_gene_synonyms/Input/Ensembl_Gene_Names_&_Synonyms_v2.csv'

#%%

## Open csv file
Anophthalmia= pd.read_excel(file_path, sheet_name='Summary_Anophthalmia', header=(0))

Coloboma= pd.read_excel(file_path, sheet_name='Summary_Coloboma', header=(0))

Microphthalmia= pd.read_excel(file_path, sheet_name='Summary_Microphthalmia', header=(0))

Human_gene_synonyms_list= pd.read_csv(Human_gene_synonyms, header=(0), on_bad_lines='skip')


#%% Remove wrong/ old gene synonyms to prevent confusion
short_list = Human_gene_synonyms_list

# #   Replace any emtpy strongs in the columns with np.nan objects
# short_list['Gene name'].replace('', np.nan, inplace=True)
# short_list['Gene Synonym'].replace('', np.nan, inplace=True)

#   Drop the null values:
short_list.dropna(subset=['Gene name'], inplace=True)    # after dropping nans, the number of rows should be 85190 gene names.

short_list.dropna(subset=['Gene Synonym'], inplace=True)    # after dropping nans, the number of rows should be 85190 gene names.

# Human_gene_synonyms = 'C:/Users/Hande/Downloads/Book4.csv'
# Human_gene_synonyms_list= pd.read_csv(Human_gene_synonyms, header=(0), on_bad_lines='skip')

temp_returning= pd.DataFrame()


for index, gene in enumerate(short_list['Gene name']): 
   print (gene)
  # print(index)    
   
   temp_return= short_list.loc[short_list['Gene Synonym'].str.contains(gene, case=True)]

   temp_returning= pd.concat([temp_returning, temp_return])
      
temp_returning = temp_returning.drop_duplicates()
returning_indexes= temp_returning.index.values.tolist()
returning_genes = temp_returning['Gene name'].tolist()

temp_keeping= Human_gene_synonyms_list.drop(returning_indexes)
   

z = Human_gene_synonyms_list

n=0
for gene in returning_genes:

    z['Gene Synonym'] = z['Gene Synonym'].replace([gene], '')
    print('Gene synonym '  + str(n) + ' of ' + str(len( returning_genes)) + ' replaced : ' + gene )
    n= (n+ 1)

gene_counts = z.groupby('Gene name').size()
gene_counts= pd.DataFrame(gene_counts)
gene_counts.columns =['Count']
gene_counts['index']= range(1, len(gene_counts) + 1)
gene_counts['Gene name']= gene_counts.index
gene_counts.set_index('index', inplace=True)

gene_counts = gene_counts[['Gene name', 'Count']]
gene_counts = gene_counts.drop(gene_counts[gene_counts.Count ==1].index)


indexes_todelete= pd.DataFrame()    
for gene in gene_counts['Gene name']:
    index = z[ (z['Gene name']== gene) & (z['Gene Synonym']== '') ].index
    index = pd.DataFrame(index)
    indexes_todelete= pd.concat([indexes_todelete, index ])
    
    
indexes_todelete= indexes_todelete.squeeze()
z= z.drop(indexes_todelete)
z= z.sort_values(by='Gene stable ID', ascending=False)
z= z.drop_duplicates(subset=['Gene name', 'Gene Synonym'], keep='last')
     
    
final_df = z
 
output_folder= (base_folder + '/Files/Removing_incorrect_gene_synonyms/Output'  ) 
final_df.to_csv((output_folder + '/Human_gene_synonyms_corrected_v2.csv'), index=False)














