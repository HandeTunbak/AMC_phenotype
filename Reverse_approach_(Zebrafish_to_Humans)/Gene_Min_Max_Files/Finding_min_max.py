# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 19:16:22 2023

@author: Dr Hande Tunbak
"""

''' 
This code searches through excel sheet with gene names and retruns two excel sheets.
The first sheet is holds positions of the gene (including al its trasncripts). 
The second sheet is the positions +-1Mb (downstream and upstream of gene position). 

'''

## Run code in command list if using for the first time  
#  >> pip install pyensembl >> in command line
## PyEnsembl currently only supports human annotations. Future releases will support other species by specifying a species argument to EnsemblRelease (e.g. EnsemblRelease(release=78, species="mouse")).
## by Ensembl install --release 108 --species homo_sapiens OR by Ensembl install --release 78 --species homo_sapiens ## 
## https://www.hammerlab.org/2015/02/04/exploring-the-genome-with-ensembl-and-python/


# Import libraries
import pandas as pd
import pyensembl

import os
import glob


## ----------------------------------------------------------------
# Set path where excel sheet is located
folder_path = os.getcwd()
## ----------------------------------------------------------------

ensembl = pyensembl.EnsemblRelease() # by default newer release is used when not specified. 

# Get all gene ids
gene_ids = ensembl.gene_ids()

# Get gene information
genes = [ensembl.gene_by_id(gene_id) for gene_id in gene_ids]

# Create data frame of genes list from ensembl
genes_df= pd.DataFrame(genes)
genes_df.columns= ["genome"]


#%% Prepare data for analysis

# Transpose human_geneome_data from horiztonal to vertical   
human_genome_data= pd.DataFrame([genes_df.genome[:]]).T

# Split a single column into two columns using apply()
human_genome_data[['Gene_id_(Ensembl_ID)', 
                   'Gene_Name', 
                   'Biotype', 
                   'Contig_(Chromosome_No.)', 
                   'Gene_start_position', 
                   'Gene_end_position', 
                   'Strand', 
                   'genome_number']] = human_genome_data["genome"].apply(lambda x: pd.Series(str(x).split(",")))


# Drop the first column with combined data for cleaner handling
human_genome_data= human_genome_data.drop(columns=['genome'])

# Remove terms from each column for easier viewing of data
human_genome_data['Gene_id_(Ensembl_ID)']= human_genome_data['Gene_id_(Ensembl_ID)'].str.replace("gene_id='" , "", regex=True)
human_genome_data['Gene_id_(Ensembl_ID)']= human_genome_data['Gene_id_(Ensembl_ID)'].str.replace("Gene", "", regex=True)
human_genome_data['Gene_id_(Ensembl_ID)']= human_genome_data['Gene_id_(Ensembl_ID)'].str.replace("(", "", regex=True)
human_genome_data['Gene_id_(Ensembl_ID)']= human_genome_data['Gene_id_(Ensembl_ID)'].str.replace("'", "", regex=True)

human_genome_data['Gene_Name']= human_genome_data['Gene_Name'].str.replace("gene_name='", "") 
human_genome_data['Gene_Name']= human_genome_data['Gene_Name'].str.replace("'", "")
human_genome_data['Gene_Name']= human_genome_data['Gene_Name'].str.replace(" ", "") # NB: there is a space before gene_name=', so this also needs to be removed

human_genome_data['Biotype']= human_genome_data['Biotype'].str.replace("biotype='", "")
human_genome_data['Biotype']= human_genome_data['Biotype'].str.replace("'", "")

human_genome_data['Contig_(Chromosome_No.)']= human_genome_data['Contig_(Chromosome_No.)'].str.replace("contig='", "")
human_genome_data['Contig_(Chromosome_No.)']= human_genome_data['Contig_(Chromosome_No.)'].str.replace("'", "")
human_genome_data['Contig_(Chromosome_No.)']= human_genome_data['Contig_(Chromosome_No.)'].str.replace(" ", "")

human_genome_data['Gene_start_position']= human_genome_data['Gene_start_position'].str.replace("start=", "")
human_genome_data['Gene_start_position']= human_genome_data['Gene_start_position'].str.replace(" ", "")

human_genome_data['Gene_end_position']= human_genome_data['Gene_end_position'].str.replace("end=", "")
human_genome_data['Gene_end_position']= human_genome_data['Gene_end_position'].str.replace(" ", "")

human_genome_data['Strand']= human_genome_data['Strand'].str.replace("strand='", "")
human_genome_data['Strand']= human_genome_data['Strand'].str.replace("'", "")
human_genome_data['Strand']= human_genome_data['Strand'].str.replace(" ", "")

human_genome_data['genome_number']= human_genome_data['genome_number'].str.replace("genome='", "")
human_genome_data['genome_number']= human_genome_data['genome_number'].str.replace("'", "")
human_genome_data['genome_number']= human_genome_data['genome_number'].str.replace(")", "", regex=True)
human_genome_data['genome_number']= human_genome_data['genome_number'].str.replace(" ", "")


#%% Load quiery genes list 

# Search through the folder and located the excel sheet
for filename in glob.glob(os.path.join(folder_path, 'Input_Quiery_Gene_List.xlsx')):

    # Print the file name with path
    print(filename)
    
    # Read excel sheet
    # NB: If the file is open on excel, reading the file with python will be deinied, therefore ensure the excel file is closed.
    Quiery_gene_list = pd.read_excel(filename)
    
    # Sort gene list in alphabetical order (a-Z)
    Quiery_gene_list= Quiery_gene_list.sort_values(by=['Gene list'], ascending = True)
    
    # Remove duplicates from list
    Quiery_gene_list= Quiery_gene_list.drop_duplicates(keep=False)
    

#%% Search for quiery gene list in human_genome_data

# Create emtpy dataframe to store values
quiery_results= pd.DataFrame()
a= human_genome_data

for gene in Quiery_gene_list['Gene list']:
    print (gene)
    
    result = human_genome_data.loc[human_genome_data['Gene_Name'] == gene]
    quiery_results= pd.concat([quiery_results,result], axis=0)
    
    
#%% Modifications (e.g., formatting) and additions to results table

# Remove columns from table    
quiery_results_final= quiery_results.drop(columns=['Biotype', 'genome_number'])
# Optional --> removal of ensembl ID
quiery_results_final= quiery_results_final.drop(columns=['Gene_id_(Ensembl_ID)'])

# Convert columns containing only numbers from str to int type https://sparkbyexamples.com/pandas/pandas-convert-column-to-int/#:~:text=to%20int%20(Integer)-,Use%20pandas%20DataFrame.,int64%20%2C%20numpy.
quiery_results_final = quiery_results_final.astype({'Gene_start_position' : 'int'})
quiery_results_final = quiery_results_final.astype({'Gene_end_position' : 'int'})

    
#%% Save results 

# Reorder columns for BedFile format before saving --> gene chromsome no., gene postion start, gene position end, gene name, gene strand 

quiery_output = pd.concat([  quiery_results_final['Contig_(Chromosome_No.)'], 
                             quiery_results_final['Gene_start_position'], 
                             quiery_results_final['Gene_end_position'], 
                             quiery_results_final['Gene_Name'], 
                             quiery_results_final['Strand']   ], axis=1)

# Remname coluimns to match ucsc output
quiery_output.columns= ['#chrom', 'txStart', 'txEnd', 'name2' , 'strand' ]


# Set output folder location
output_folder = folder_path

# Save as a bed file 
quiery_output.style.set_properties(**{'text-align': 'center'}).to_excel(output_folder + '/Output_Quieried_gene_list.xlsx', index=False)
#quiery_output.to_excel(output_folder + 'Output_Quieried_gene_list.xlsx',index=False )


#%% Modifiy table of results to include 1Mb upstream and downstream of gene position

# # Add columns 
# # column for start position -1Mb 
# quiery_results_final['Gene_start_position_-1Mb']= quiery_results_final['Gene_start_position'] - 1000000

# # column for end position +1Mb 
# quiery_results_final['Gene_end_position_-1Mb']= quiery_results_final['Gene_end_position'] + 1000000

# column for start position -1Mb 
quiery_output['txStart']= quiery_output['txStart'] - 1000000
quiery_output.rename(columns = {'txStart':'txStart -1MB'}, inplace = True)
    
# column for end position +1Mb 
quiery_output['txEnd']= quiery_output['txEnd'] + 1000000
quiery_output.rename(columns = {'txEnd':'txEnd +1MB'}, inplace = True)
    
  
#%% Save results with downstream and upstream positions
    
# Save as a bed file 
quiery_output.style.set_properties(**{'text-align': 'center'}).to_excel(output_folder + '/Output_Quieried_gene_list_expanded_1Mb.xlsx', index=False)


#%% Finish

## END

    


