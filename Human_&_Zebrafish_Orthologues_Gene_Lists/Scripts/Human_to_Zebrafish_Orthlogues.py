# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Dr Hande Tunbak)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import time
 
folder_path= 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/Useful_data_resources/Human_&_Zebrafish_Orthologues_Gene_Lists/Human_to_Zebrafish'
file_path= 'C:/Users/hande/OneDrive - Oxford Brookes University/Documents/Projects/Useful_data_resources/Human_&_Zebrafish_Orthologues_Gene_Lists/Human_to_Zebrafish/Ensembl_Orthologues_Human_to_Zebrafish_Gene_Names.xlsx'

file= pd.read_excel(file_path, header=(0))
file.columns= ['Human_Gene_Name', 'Zebrafish_Gene_Name 0']

i=1
for interations in range(1, 45): 
    print(interations)
        
    column_name = str( 'Zebrafish_Gene_Name ' + str(interations))
    file[column_name]=np.nan  
    
    rows_to_drop = []
    

    for currentindex in range(i, len(file)-1):
        #print (currentindex)
        previous_index= currentindex -1
        previous_column_name =str( 'Zebrafish_Gene_Name ' + str(interations -1))
        
      
        if file['Human_Gene_Name'][currentindex]== file['Human_Gene_Name'][previous_index]:
    
            a = file[previous_column_name][currentindex]
            file[column_name][currentindex] =  file[previous_column_name][previous_index]
            
            rows_to_drop.append(previous_index)
            
           
        else:
            pass
        
        print('Completed interation : ' + str(interations) + ' index : ' + str(currentindex)  )
    i=+1    
#file.drop(labels=next_index, axis=0)
        

file_to_save = file.drop(labels=rows_to_drop, axis=0)

file_to_save.to_csv((folder_path + '/Human_to_Zebrafish_Orthologues.csv'), index=False)