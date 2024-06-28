#%% # Do Stats for Average Eye Length
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# Create report file
reportFilename = ( Experiment_folder + '/Figures_and_Stats' + 
                  r'/Stats_Report_Average_Eye_Length.txt')
reportFile = open(reportFilename, 'w')

#-----------
# Average Eye Lengths: Control and Injected
s1= Control_MAB_siblings_mean_eye_lengths   * Eye_conversion_8x
s2= Control_MAB_phenos_mean_eye_lengths     * Eye_conversion_8x
s3= Injected_MAB_siblings_mean_eye_lengths  * Eye_conversion_8x
s4= Injected_MAB_phenos_mean_eye_lengths    * Eye_conversion_8x
  
reportFile.write( '\n' + '#---------------------------' + '\n' )
reportFile.write('Average Eye Length: Control vs Injected' + '\n' + '\n')  

# Find means
s1_mean = np.mean(s1)
s2_mean = np.mean(s2)
s3_mean = np.mean(s3)
s4_mean = np.mean(s4)

reportFile.write('Mean Control MAB siblings = ' + str(s1_mean) + '\n')
reportFile.write('Mean Control MAB phenos= ' + str(s2_mean) + '\n')
reportFile.write('Mean Injected MAB siblings = ' + str(s3_mean) + '\n')
reportFile.write('Mean Injected phenos siblings = '  + str(s4_mean) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

## --------------
## Mean Control MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s3)
reportFile.write( 'Means: Control MAB siblings vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s3)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB pheno vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s2, s4)
reportFile.write( 'Means: Control MAB pheno vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')


# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s2, s4)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n')

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB siblings vs Mean Control MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s1, s2)
reportFile.write( 'Means: Control MAB siblings vs Control MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s1, s2)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n' )

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Injected MAB siblings vs Mean Injected MAB phenos
# Unpaired t-test
t, p = sp.stats.ttest_ind(s3, s4)
reportFile.write( 'Means: Injected MAB siblings vs Injected MAB phenos' + 
                 '\n' + 'Unpaired t-test' +'\n' +
                 't= ' + str(t) +'\n' +
                 'p= ' + str(p) + '\n')

# NonParametric Mannwhitney U-test
u, p = sp.stats.mannwhitneyu(s3, s4)
reportFile.write('\n' + 'MannWhiteney U-test' +'\n' +
                 'u= ' + str(u) +'\n' +
                 'p= ' + str(p) + '\n' + '\n' )

reportFile.write('#------------------------' + '\n')

#-----------------
## Mean Control MAB siblings vs Mean Injected MAB siblings
percentage_change= (np.mean(s1) - np.mean(s3) / np.mean(s1) ) *100 # in um
reportFile.write( 'Percentage  Difference: ' + str(percentage_change) +  '(Mean Control MAB siblings ->  Mean Injected MAB sibllings)' + '\n' )

## Mean Control MAB pheno vs Mean Injected MAB phenos
percentage_change= (np.mean(s2) - np.mean(s4) / np.mean(s2) ) *100 # in um
reportFile.write( 'Percentage  Difference: ' + str(percentage_change) +  '(Mean Control MAB phneos ->  Mean Injected MAB phenos)' + '\n' )


#-----------------
#------------------------
reportFile.close()
