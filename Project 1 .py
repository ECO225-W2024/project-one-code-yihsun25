#!/usr/bin/env python
# coding: utf-8

# # Investigate the combined influence of neighborhood groups and room types on the pricing strategies of Airbnb listings in New York City
# 
# ## Introduction 
# 
# ###

# ## Data Cleaning/Loading

# In[19]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd 


# In[18]:


nRowsRead = 1000
# AB_NYC_2019.csv has 48895 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('AB_NYC_2019.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'AB_NYC_2019.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[20]:


df1.head(5)


# ## Summary Statistics Table

# In[16]:


df1.describe()


# In[3]:


df.describe().head()


# ## Plots, Histograms, Figures 

# In[ ]:





# ## Conclusion

# In[ ]:




