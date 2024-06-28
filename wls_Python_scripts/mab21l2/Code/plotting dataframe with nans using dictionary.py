# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:02:13 2024

@author: hande
"""


#%%

import os
import sys
import glob
import cv2
import numpy as np
import scipy as sp
import imageio
import matplotlib.pyplot as plt
import pandas as pd
import math
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path
import seaborn as sns
from scipy import stats

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "svg"

from skimage import io, color, data, filters, measure, morphology, img_as_float, exposure, restoration
from skimage.color import rgb2gray
from skimage.filters import unsharp_mask, threshold_multiotsu, threshold_isodata, threshold_otsu, threshold_minimum, threshold_yen
from skimage.filters.rank import autolevel, enhance_contrast
from skimage.morphology import disk, diamond,  white_tophat, binary_dilation, remove_small_objects, label

#%%


a= [1,2,3]
b= [1,2,3]
c= [1,2,3]
d= [1,2]


df = pd.DataFrame({
    'Symbol': ['A', 'B', 'C', 'D'],
'Info': [a, b, c, d] })

df= df.explode('Info')
plt.Figure()

ax=sns.barplot(  x=df['Symbol'] , y=df['Info'] , data=df, estimator='mean', errorbar=('se'), capsize=.2, errwidth=2.1, color='lightblue', width=0.45) 
ax.spines[['right', 'top']].set_visible(False)

# df_copy= df

# df_copy['Sentiment'] = df_copy['Mentions'].apply(lambda x:  - 1)



