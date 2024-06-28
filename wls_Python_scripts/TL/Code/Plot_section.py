#%% # Plot Average Eye Area
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

s1= Control_MAB_siblings_avg_eye_area
s2= Control_MAB_phenotps_avg_eye_area
s3= Injected_MAB_siblings_avg_eye_area
s4= Injected_MAB_phenotps_avg_eye_area

df1= pd.DataFrame({
    'Condition': ['Control MAB sibling', 'Control MAB phenos', 'Injected MAB sibling', 'Injected MAB phenos'], 
    'Area': [ s1, 
                 s2, 
                 s3, 
                 s4    ] })

df1=df1.explode('Area')

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(x= df1['Condition'], y= df1['Area'], data=df1, estimator='mean', errorbar=('se'), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Eye Area (µm2)', fontsize=10 )
plt.title(' Average Eye Length: Control vs Injected')
plt.ylim(2600, 3600)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Area', dpi=900)  

plt.show()

#-------------------------
# # Plot Figure
plt.Figure()

ax=sns.barplot(x= df1['Condition'], y= df1['Area'], data=df1, estimator='mean', errorbar=('se'), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

plt.xlabel('Condition', fontsize=10)
plt.ylabel('Eye Area (µm)', fontsize=10 )
plt.title(' Average Eye Length: Control vs Injected')
#plt.ylim(2600, 3600)

plt.tight_layout()

# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Area_Default', dpi=900)  

plt.show()


#------------------
# # Plot Boxplot with Scatter plot

s1= Control_MAB_siblings_avg_eye_area
s1a= pd.DataFrame([s1]).T
s1a.columns=['Control MAB siblings']
s1a = s1a.set_index(np.arange(0,len(s1a)))


s2= Control_MAB_phenotps_avg_eye_area
s2a= pd.DataFrame([s2]).T
s2a.columns=['Control MAB phenos']
s2a = s2a.set_index(np.arange(0,len(s2a)))


s3= Injected_MAB_siblings_avg_eye_area
s3a= pd.DataFrame([s3]).T
s3a.columns=['Injected MAB siblings']
s3a = s3a.set_index(np.arange(0,len(s3a)))


s4= Injected_MAB_phenotps_avg_eye_area
s4a= pd.DataFrame([s4]).T
s4a.columns=['Injected MAB phenos']
s4a = s4a.set_index(np.arange(0,len(s4a)))



df1= pd.DataFrame({
    'Condition': ['Control MAB sibling', 'Control MAB phenos', 'Injected MAB sibling', 'Injected MAB phenos'], 
    'Area': [ s1, 
                 s2, 
                 s3, 
                 s4    ] })


plt.figure(1)
  
ax= plt.subplot()

ax.boxplot(df1['Area'], labels= df1['Condition'],   showfliers=False,   widths=0.25, positions=range(len(df1['Condition'])))

df1=df1.explode('Area')
ax.scatter(y=df1['Area'], x=df1['Condition'], c= 'lightblue' , alpha=1.0)

plt.title('Average Eye Lengths')
plt.tight_layout()

ax.spines[['right', 'top']].set_visible(False)

    
# Save figure
plt.savefig( Experiment_folder + '/Figures_and_Stats' + '/Average_Eye_Area-Boxplot_with_scatter', dpi=900)  

plt.show()    