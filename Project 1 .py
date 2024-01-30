#!/usr/bin/env python
# coding: utf-8

# # Investigate the joint influence of neighborhood groups, neighborhood and room types on the pricing strategies of Airbnb listings in New York City
# 
# ## Introduction 
# 
# #### <span style="color:blue">Independent variables:</span>
#     1. Neighbourhood group (e.g., Mahattan, Brooklyn, Queens, Staten Island, Bronx) 
# New York City's neighborhood groups, like Manhattan, Brooklyn, Queens, etc., are known to have distinct characteristics in terms of their cultural, economic, and social profiles. These differences can significantly impact the demand and perceived value of Airbnb listings in these areas. Prices in real estate and lodging often vary dramatically from one neighborhood group to another due to factors like desirability, accessibility, and local amenities. By analyzing different neighborhood groups, you can identify broader geographical trends in Airbnb pricing.
# 
#     2. Neighbourhood (e.g.,Kensington, Midtown, Harlem) -total of 221 
# Even within a single neighborhood group, specific neighborhoods can have their unique appeal or drawbacks, affecting the pricing of Airbnb listings. For instance, a neighborhood's safety, proximity to tourist attractions, or nightlife can play a crucial role.This variable allows a more granular analysis than just looking at neighborhood groups. It can reveal hyper-local trends and anomalies in pricing that might not be apparent at the neighborhood group level.
#     
#     3. Room type (e.g., Private room, Entire home/apt, Shared room) 
# Different room types cater to different types of travelers and needs. For example, entire homes/apartments are likely more expensive than private or shared rooms, reflecting different levels of privacy, space, and amenities. Understanding how room type affects pricing can help identify what type of listings are more lucrative and in demand in various parts of the city. This is crucial for analyzing the market dynamics from both a host’s and a guest's perspective.
# 
# #### <span style="color:blue">Dependent variable:</span>
#     1. Price (in dollars) 
# As the primary output of interest, the price reflects the monetary value placed on an Airbnb listing, encapsulating factors like demand, location quality, and type of accommodation.

# ## Data Cleaning/Loading

# In[30]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd 


# In[38]:


df = pd.read_csv('AB_NYC_2019.csv', delimiter=',')
df.dataframeName = 'AB_NYC_2019.csv'
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[39]:


df.head(5)


# In[40]:


print(df.dtypes) #identify the type of each columns in the dataset 


# In[41]:


df.isnull().any(axis=0)


# In[42]:


# drop all rows containing a missing observation
df.dropna()


# In[47]:


num_unique_neighbourhoods = df['neighbourhood'].nunique()
print(num_unique_neighbourhoods)


# In[50]:


num_unique_neighbourhoods_group = df['neighbourhood_group'].unique()
print(num_unique_neighbourhoods_group)


# In[51]:


num_unique_room_type = df['room_type'].unique()
print(num_unique_room_type)


# ## Summary Statistics Table

# In[43]:


df.describe()


# In[44]:


df.describe().head()


# In[61]:


# Select the variables of interest
variables = ['neighbourhood_group', 'neighbourhood', 'room_type', 'price']  

# Use a loop to generate summary statistics for each variable
for var in variables:
    print(f"Summary statistics for {var}:")
    print(df[var].describe(include='all'))
    print("\n")  # Adds a newline for better readability


# ## Plots, Histograms, Figures 

# In[ ]:





# ## Conclusion

# In[ ]:




