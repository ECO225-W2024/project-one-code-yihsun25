#!/usr/bin/env python
# coding: utf-8

# # How do neighborhood characteristics and accomodation type influence the pricing of Airbnb listings in New York City in 2019?"
# 
# ## Introduction 
# 
# The rise of Airbnb has notably transformed urban accommodation landscapes, particularly in global cities like New York City (NYC). NYC, with its diverse neighborhoods and pivotal role as a tourism and business nexus, serves as an ideal backdrop for examining the impact of short-term rentals on local housing markets and neighborhood dynamics. This study zeroes in on the year 2019, a critical period before the COVID-19 pandemic, providing a vital baseline to understand the interplay between Airbnb listing prices, neighborhood characteristics, and room types.
# 
# At the heart of this investigation is the question: **"How do neighborhood characteristics and room type influence the pricing of Airbnb listings in New York City in 2019?"** This query underscores the hypothesis that neighborhood features (e.g., safety, accessibility, cultural attractions) and the nature of the room offered (entire home/apt, private room, shared room) are significant determinants of listing prices. The research leverages a detailed 2019 Airbnb dataset (AB_NYC_2019), which includes prices, room types, host, review metrics, and availability, to explore how these factors collectively shape pricing strategies. The source of this dataset is Inside Airbnb, an investigatory website found by Murray Cox in 2016.
# 
# This study aims to illuminate the pricing dynamics within the Airbnb market, offering insights for hosts, guests, and policymakers. By analyzing the nuanced interactions between neighborhood appeal and accommodation type, the research provides a comprehensive view of the factors driving short-term rental pricing in NYC. The findings aim to inform pricing strategies, enhance guest decision-making, and guide policy discussions on the sharing economy's role in urban settings, thereby contributing to a broader understanding of the economic and social implications of platforms like Airbnb.
# 
# #### <span style="color:blue">Independent variables(X):</span>
# 1. **Neighbourhood group** (e.g., Mahattan, Brooklyn, Queens, Staten Island, Bronx) <br>
# New York City's neighborhood groups are known to have distinct characteristics in terms of their cultural, economic, and social profiles. These differences can significantly impact the demand and perceived value of Airbnb listings in these areas. Prices in real estate and lodging often vary dramatically from one neighborhood group to another.
# 
# 2.  **Neighbourhood** (e.g.,Kensington, Midtown, Harlem, etc) <br>
# Even within a single neighborhood group, specific neighborhoods can have their unique appeal or drawbacks, affecting the pricing of Airbnb listings. For instance, a neighborhood's safety, proximity to tourist attractions, or nightlife can play a crucial role.This variable allows a more granular analysis than just looking at neighborhood groups. It can reveal hyper-local trends and anomalies in pricing that might not be apparent at the neighborhood group level.
# 
# 3. **Room type** (e.g., Private room, Entire home/apt, Shared room) <br>
# Different room types cater to different types of travelers and needs. For example, entire homes/apartments are likely more expensive than private or shared rooms, reflecting different levels of privacy, space, and amenities. Understanding how room type affects pricing can help identify what type of listings are more lucrative and in demand in various parts of the city. This is crucial for analyzing the market dynamics from both a host’s and a guest's perspective.
# 
# 4. **Numer of Reviews** <br>
# The number of reviews can be seen as an indicator of a listing's popularity and the level of trust guests place in it. Listings with more reviews may be perceived as more popular or trustworthy, affecting their pricing power.
# 
# #### <span style="color:blue">Dependent variable (Y):</span>
# 1. **Price (in dollars)** <br>
# As the primary output of interest, the price reflects the monetary value placed on an Airbnb listing, encapsulating factors like demand, location quality, and type of accommodation. The volume of reviews can provide potential guests with more information to assess the quality of a listing. This perceived quality can influence what guests are willing to pay.When analyzed alongside variables like neighborhood groups, specific neighborhoods, and room types, the number of reviews adds another dimension to understand the pricing dynamics.

# ## Data Cleaning/Loading

# In[54]:


#Import libraries
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.colors as mplc
import matplotlib.patches as patches
import matplotlib.pyplot as plt 
import plotly.express as px
import seaborn as sns
import numpy as np 
import os 
import pandas as pd 


# In[55]:


#Load Dataset 
df = pd.read_csv('AB_NYC_2019.csv', delimiter=',')
df.dataframeName = 'AB_NYC_2019.csv'


# In[4]:


nRow, nCol = df.shape 
print(f'There are {nRow} rows and {nCol} columns')


# In[5]:


df.head(5)


# In[6]:


dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])

dtypes_df 


# In[8]:


Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
#price mean = 152.72
price_outliers = df[(df['price'] < (Q1 - 1.5 * IQR)) | (df['price'] > (Q3 + 1.5 * IQR))]
price_outliers


# - The dataset contains 2,972 observations, among which the price is identified as an outlier.

# In[9]:


Q1 = df['number_of_reviews'].quantile(0.25)
Q3 = df['number_of_reviews'].quantile(0.75)
IQR = Q3 - Q1

num_review_outliers = df[(df['number_of_reviews'] < (Q1 - 1.5 * IQR)) | (df['number_of_reviews'] > (Q3 + 1.5 * IQR))]
num_review_outliers


# - The dataset contains 6021 observations, among which the number of reviews is identified as an outlier.

# In[10]:


missing_values = df.isnull().any(axis=0)
missing_values_df = pd.DataFrame(missing_values, columns=['Has Missing Values'])

missing_values_df 


# In[108]:


num_null = df.isnull().sum()
num_null  = pd.DataFrame(num_null , columns=['Num of Missing Value'])

num_null 


# In[201]:


df1 = df.dropna()
df1


# - df1 is the new data frame that dropped observations with missing values. df1 will not be used in Project 1 as there are no missing values in the chosen independent variables. 

# In[123]:


df.duplicated().sum() 


# - No duplicate entries are present in the dataset.

# ## Summary Statistics Table

# In[220]:


variables = ['neighbourhood_group', 'neighbourhood', 'room_type', 'number_of_reviews', 'price']

summary_df = pd.DataFrame()

for var in variables:
    description = df[var].describe(include='all').to_frame().transpose()
    description.index = [var]
    summary_df = pd.concat([summary_df, description], axis=0)
    
summary_df


# **Count**: There are 48,895 entries or listings in the dataset
# <br>
# <br>
# **Neibourhood group (X1)**
# <br> 
# > - **Unique:** The data set focuses on 5 unique neighbourhood groups.   <br> 
# > - **Top/Frequency:** The most common neighbourhood group is Manhattan, with 21,661 listings. This suggests that Manhattan is the most listed area in the dataset. 
# <br>
# 
# **Neighbourhood (X2)** 
# <br> 
# > - **Unique:** There are 221 unique neighbourhoods. This suggests a very detailed level of data granularity, allowing for an in-depth analysis of location-based pricing strategies. <br> 
# >  - **Top/Frequency:** Williamsburg is the most common neighbourhood with 3,920 listings. This could mean that Williamsburg is a popular area for Airbnb listings or it could reflect a specific trend or preference among Airbnb hosts or guests in New York City.
# <br>
# 
# **Room Type (X3)** <br> 
# > - **Unique:** There are 3 unique room types. This limited number of categories can make it easier to analyze and compare the impact of room type on pricing. <br> 
# > - **Top/Frequency:** The most common room type is "Entire home/apt" with 25,409 listings. This prevalence suggests that entire homes or apartments are a popular choice on Airbnb. 

# **Number of Reviews (X4)** 
# > - **Mean:** On average, a listing has about 23.27 reviews. <br>
# > - **Standard Deviation (std):** The high standard deviation of approximately 44.55 suggests that the number of reviews per listing varies widely.<br>
# > - **Minimum:** The minimum number of reviews is 0, indicating that there are some listings without any reviews.<br>
# > - *25th Percentile:** 25% of the listings have 1 or fewer reviews, showing a significant number of listings with very few reviews. <br>
# > - **Median (50%):** The median value is 5 reviews, which means that half of the listings have more than 5 reviews and half have fewer. The median is much lower than the mean, suggesting that the distribution of reviews is right-skewed with some listings having a very high number of reviews.<br>
# > - **75th Percentile:**  75% of listings have 24 or fewer reviews. This further indicates the presence of listings with a very high number of reviews, which could be outliers influencing the mean.<br>
# > - **Maximum:** The maximum number of reviews is 629, which is much higher than the mean and median, confirming the right-skewed distribution.
# <br>
# <br>
# 
# **Price (Y)**
# > - **Mean:** The average price for a listing is approximately \\$152.72. This is a starting point for understanding the typical cost but doesn't capture the full picture of pricing. <br>
# > - **Standard Deviation (std):** The standard deviation is very high at approximately \\$240.15, indicating a large variation in pricing among listings. <br>
# > - **Minimum:** The minimum price is \\$0, which might indicate free listings, a data entry error, or special promotions. <br>
# > - **25th Percentile:** The 25th percentile is \\$69, indicating that a quarter of the listings are priced at \\$69 or less. <br>
# > - **Median (50%):** The median price is \\$106, which, like with reviews, is lower than the mean, suggesting that there are some very high-priced listings that are pulling the average up. <br>
# > - **75th Percentile:** At the 75th percentile, listings are priced at \\$175 or less. <br>
# > - **Maximum:** The maximum price is \\$10,000, which is exceptionally high compared to the rest of the data and likely represents luxury or unique listings.

# ## Plots, Histograms, Figures 

# In[106]:


sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='neighbourhood_group', data=df, palette='muted', ax=ax)

ax.set_title('Distribution of Listings Across Neighborhood Groups')
ax.set_xlabel('Neighborhood Group')
ax.set_ylabel('Frequency')

plt.show()


# - The observed pattern indicates that Airbnb activity is not evenly distributed across New York City's boroughs, with a clear preference for Brooklyn and Manhattan. This uneven distribution suggests that these boroughs are likely to exhibit distinct pricing behaviors due to their higher demand and popularity, which is crucial for analyzing the impact of neighborhood characteristics on pricing in the Airbnb market.

# In[249]:


counts = df['neighbourhood'].value_counts()

# Keep only the top N categories and group the rest as 'Other'
N = 10
top_n = counts[:N].index
df['neighbourhood_reduced'] = df['neighbourhood'].where(df['neighbourhood'].isin(top_n), 'Other')

plt.figure(figsize=(12, 6))
sns.countplot(y='neighbourhood_reduced', data=df, order=df['neighbourhood_reduced'].value_counts().index, color='skyblue')

plt.title('Top 10 Neighborhoods Distribution')
plt.xlabel('Frequency')
plt.ylabel('Neighborhood')

plt.show()


# - The dataset encompasses 221 neighborhoods; to effectively visualize this, a 'Top 10' strategy was employed, showcasing the neighborhoods with the highest number of Airbnb listings in NYC. Williamsburg and Bedford-Stuyvesant emerge as the neighborhoods with a notably higher concentration of listings, while the remaining listings are more evenly distributed among the other neighborhoods.

# In[107]:


sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='room_type', data=df, palette='muted', ax=ax)

ax.set_title('Distribution of Listings Across Room Types')
ax.set_xlabel('Room Type')
ax.set_ylabel('Frequency')

plt.show()


# - The countplot reveals a dominant trend in NYC's Airbnb listings, heavily favoring entire homes/apartments and private rooms over shared rooms. This preference reflects market demand for more private and spacious accommodations, likely influencing higher price points for these types of listings. Such a distribution is crucial for assessing the impact of room type on Airbnb pricing within the city's competitive lodging market.

# In[18]:


sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(df['number_of_reviews'], bins=50, kde=False, color='skyblue')

plt.title('Distribution of Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')
plt.xlim(0, 300) #Limited range of x-axis to better illustrate the shape of the distribution 

plt.show()


# - To enhance the readability of the histogram illustrating the distribution of review counts for Airbnb listings, which exhibits a right-skewed pattern indicating that the majority of listings have a low number of reviews, with only a few listings receiving a large number of reviews, I limited the range of the data. This approach allows for a clearer visualization of the data distribution, enabling us to still observe the overall trend without the distortion caused by extreme values. This adjustment is particularly useful in highlighting the variability in listing popularity or engagement. Listings that accumulate a higher number of reviews may be able to command premium pricing, reflecting increased demand often driven by factors such as desirable neighborhood characteristics or preferred accommodation types. Such trends underline the importance of location desirability and the nature of the accommodation in determining Airbnb pricing strategies in New York City.

# In[114]:


sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=100, kde=False, color='skyblue')

plt.title('Distribution of Airbnb Listing Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Frequency')
plt.xlim(0, 1500)

plt.show()


# - To increase the readability of the plot and better illustrate the distribution of prices for NYC Airbnb listings, the data range has been limited and outliers removed. This methodological adjustment, crucial for enhancing the clarity of the histogram, ensures that the positively skewed nature of the price distribution is preserved. It effectively highlights that while the market is predominantly composed of affordable options, there is a notable presence of premium-priced listings. This variation in pricing is largely influenced by factors such as desirable locations, superior room types, or luxury amenities, which are key to understanding the pricing dynamics within the Airbnb market in New York City.

# In[297]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='neighbourhood_group', y='price', data=df)

plt.title('Price Distribution Across Neighborhood Groups (Unfiltered)')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Price')

plt.show()


# In[105]:


upper_limit = df['price'].quantile(0.95)
filtered_df = df[df['price'] <= upper_limit]

plt.figure(figsize=(12, 6))
sns.boxplot(x='neighbourhood_group', y='price', data=filtered_df)

plt.title('Price Distribution Across Neighborhood Groups (Filtered Outliers)')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Price')
plt.show()


# - The box plot, refined by excluding data above the 95th percentile, presents a clearer view of Airbnb pricing trends across New York City neighborhoods, highlighting typical price ranges while controlling for extreme outliers. Manhattan emerges with the highest median prices and a wide interquartile range, reflecting substantial price variability and indicating a market that can sustain higher rental costs, possibly due to its prime location and attractiveness to renters. In contrast, Brooklyn shows moderately high median prices with a more compressed interquartile range, signaling less price fluctuation. Queens, Staten Island, and the Bronx exhibit lower median prices and narrow interquartile ranges, pointing to a more uniform pricing approach and suggesting that these areas may be less affected by the premium associated with location. This pattern of pricing suggests a clear correlation between neighborhood desirability and Airbnb pricing strategies, with central, popular locations commanding higher prices, thereby aligning with the research inquiry into the influence of location on accommodation pricing in urban settings.

# In[21]:


upper_limit = df['price'].quantile(0.95)
filtered_df = df[df['price'] <= upper_limit]

plt.figure(figsize=(12, 6))
sns.boxplot(x='room_type', y='price', data=filtered_df)

plt.title('Price Distribution By Room Type (Filtered Outliers)')
plt.xlabel('Room Type')
plt.ylabel('Price')
plt.show()


# - This box plot displays the price distribution for three room types. The median price for entire homes/apartments is the highest, with a wide interquartile range, indicating significant price variation within this category. Private rooms have a lower median price and a narrower interquartile range, showing less price variability. Shared rooms have the lowest median price and the smallest interquartile range, suggesting the least price variation among the three types. The information from this plot is valuable for researching pricing strategies, as it suggests that the type of room is a major determinant of price. The greater price range for entire homes/apartments implies a higher revenue potential, which could influence hosts to prefer offering their properties as entire units rather than shared spaces. This could have implications for market supply dynamics and overall pricing models within the Airbnb market in New York City.

# In[94]:


fig = px.scatter(df, x='number_of_reviews', y='price', opacity=0.5, size_max=8)
fig.update_traces(marker=dict(size=7))

fig.update_layout(
    xaxis=dict(
        tickformat=',',  
        title='Number of Reviews'
    ),
    yaxis=dict(
        tickformat='', 
        title='Price Per Night'
    ),
    title='Airbnb Price vs Number of Reviews',
    width=1000,  
    height=600  
)

fig.show()


# - The scatter plot depicts a concentration of data points in the lower left corner, indicating that the majority of Airbnb listings have a lower price range and a smaller number of reviews. The presence of data points spread vertically along the y-axis at lower review counts suggests a wide range of prices for listings with few reviews. However, both price and number of reviews exhibit extreme outliers, with some listings having a very high price per night or a very large number of reviews, which are not representative of the general pattern.
# The pattern seen in this plot can contribute to the research question by suggesting that, generally, there may not be a strong direct correlation between the number of reviews and the price per night. This could imply that factors other than customer feedback volume, such as location, room type, or amenities, might be more influential in determining the price. However, the outliers indicate exceptional cases that may warrant individual examination to understand unique pricing strategies or value propositions that lead to significant deviations from the typical price-review relationship.

# In[96]:


frac_matrix_norm = (df.groupby(['neighbourhood_group', 'room_type'])
                    .size()
                    .unstack()
                    .div(df.groupby(['neighbourhood_group']).size(), axis=0))

frac_matrix_norm.plot(kind='bar', stacked=True, figsize=(10, 6))  
plt.ylabel('Room Type Fraction')
plt.xlabel('Neighbourhood Group')
plt.legend(title='Room Type', loc='upper right')
plt.title("Fraction of Room Types Across Neighborhood Groups")


# - This matrix illustrates that entire homes/apartments constitute the majority of listings in every neighborhood, followed by private rooms. Shared rooms represent the smallest fraction in each neighborhood, with an almost negligible presence in some areas.The prevalence of entire homes/apartments being listed, particularly in Manhattan and Brooklyn, suggests a trend towards renting out entire properties, which may command higher prices compared to private or shared rooms. This observation contributes to the research question by indicating that room type is a significant factor in pricing strategies. It implies that neighborhoods with a higher fraction of entire homes/apartments could be associated with higher overall rental prices, as these types of accommodations typically offer more privacy and space, and thus, could be more in demand.

# In[100]:


dataset_original = [df[df['room_type']=='Entire home/apt']['number_of_reviews'].values,
                    df[df['room_type']=='Private room']['number_of_reviews'].values,
                    df[df['room_type']=='Shared room']['number_of_reviews'].values]

plt.violinplot(dataset = dataset_original)
plt.xticks([1, 2, 3], ['Entire home/apt', 'Private room', 'Shared room'])
plt.ylabel('Number of Reviews')
plt.xlabel('Room Type')
plt.title("Distribution of the Number of Reviews for Each Room Type")

plt.ylim(0, 300)  #limit y-axis range to increase readibility 

plt.show()


# - The distribution of reviews across room types—entire homes/apartments, private rooms, and shared rooms—is skewed towards a lower review count, with a some listings receiving a high number of reviews. To ensure clarity of the plot, the range on the y-axis was limited to exclude extreme outliers without affecting the overall skewed distribution. This pattern suggests that while room type does influence review frequency, outliers indicate that exceptional listings may disproportionately attract more reviews. These insights are valuable for understanding how the room type and guest experience impact the perceived value and popularity of Airbnb listings, a key factor in pricing strategy research.

# In[115]:


fig, ax = plt.subplots(figsize=(9, 5))
sns.stripplot(x="neighbourhood_group", y="price", hue="room_type", data=df, palette="muted", ax=ax, dodge=True)

ax.set(title='Prices Distribution by Neighbourhood Group and Room Type (Unadjusted)', xlabel='Neighbourhood Group', ylabel='Price (in dollars)')
plt.legend(title='Room Type')

plt.show()


# In[110]:


fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x="neighbourhood_group", y="price", hue="room_type", data=df, palette="muted", ax=ax)
ax.set(title='Average Price by Neighbourhood Group and Room Type', xlabel='Neighbourhood Group', ylabel='Average Price (in dollars)')

plt.show()


# - The bar chart represents the average prices for different room types within various New York City neighborhoods, calculated to mitigate the influence of extreme outliers in the dataset. Entire homes/apartments emerge as the most expensive category, with Manhattan prices being the highest, reflecting its market desirability. Private rooms are priced lower than entire apartments, yet Manhattan remains the most expensive location for this room type as well. Shared rooms stand out as the most budget-friendly option across the board, displaying minimal price fluctuation between neighborhoods.This focus on average prices clarifies the impact of both room type and neighborhood on pricing, notwithstanding the outliers. Such a presentation is crucial to discern the broader pricing trends, addressing the research question about the determinants of Airbnb pricing, with particular emphasis on the interplay between location desirability and room category.

# ## Conclusion

# The detailed examination of Airbnb listings in New York City in 2019 has yielded several findings that are pivotal to understanding the interrelations between pricing, room types, and neighborhood characteristics. The data distinctly illustrates that the type of accommodation—entire homes/apartments versus private or shared rooms—plays a crucial role in pricing. Entire homes and apartments not only command higher prices but also display a wide range of prices, particularly in Manhattan, pointing to a market that values privacy and space. Private rooms exhibit a lower median price and less variability, suggesting a more budget-conscious segment of the market, while shared rooms consistently show the lowest price points and variability, indicating a niche market with limited demand.The presence of extreme outliers in both price and the number of reviews for some listings suggests that there are exceptional properties that significantly deviate from the average, possibly due to luxury features, unique locations, or premium amenities. Moreover, the data suggests a weak or non-existent relationship between the number of reviews and the price, which could imply that guests may prioritize other factors over the quantity of reviews when making booking decisions.The distribution of room types across different neighborhoods highlights a market trend toward entire homes and apartments, especially in Manhattan and Brooklyn. This suggests a potential preference for accommodations that offer more privacy, which could be influencing the overall higher price points in these areas. Shared rooms are the least common across the board, further emphasizing the market's preference for private spaces.
# In terms of neighborhood distribution, the prevalence of listings in neighborhoods like Williamsburg and Bedford-Stuyvesant indicates a concentration of market activity that could be tied to the desirability or emerging popularity of these areas. However, the distribution of prices across neighborhoods shows that areas like Queens, Staten Island, and the Bronx tend to have lower median prices, possibly due to lower demand or less perceived desirability compared to central areas like Manhattan.
# 
# These initial observations suggest that guests may consider aspects other than reviews, such as location desirability or special amenities, when choosing accommodations. The distribution of room types suggests a preference for entire homes, especially in popular neighborhoods like Williamsburg and Bedford-Stuyvesant, while areas such as Queens, Staten Island, and the Bronx are associated with more affordable listings.To deepen the understanding of these pricing dynamics, future research should employ regression analysis to discern the precise impact of neighborhood attributes on prices and spatial analysis to visualize geographic pricing patterns. Such detailed examination will enhance the comprehension of Airbnb's complex economic ecosystem and inform strategic pricing and policy development.
