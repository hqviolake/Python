#!/usr/bin/env python
# coding: utf-8

# ### The primary objective of this project was to conduct a comprehensive analysis of the apartment real estate market in the city of Porto Alegre, Brazil and make price's prediction. To achieve this goal, we utilized the pandas, matplotlib, seaborn and sklearn libraries.

# In[1]:


import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df_vivareal = pd.read_csv('vivareal.csv')


# #### For see the data in the dataframe

# In[3]:


pd.set_option('display.max_columns', None)
df_vivareal.head(5)


# #### For see the shape of the dataframe, how many lines and columns

# In[4]:


df_vivareal.shape


# #### Deleting all lines with NaN values

# In[5]:


df_vivareal = df_vivareal.dropna(how='all')


# In[6]:


df_vivareal.shape


# #### Rename the columns name

# In[7]:


df_vivareal.rename(columns={'usableAreas/0':'area',
                          'unitTypes/0':'typeImovel',
                          'parkingSpaces/0':'garage',
                          'address/city':'city',
                          'address/state':'state',
                          'address/neighborhood':'neighborhood',
                          'address/point/lat':'latitude',
                          'address/point/lon':'longitude',
                          'suites/0':'suites',
                          'bathrooms/0':'bathroom',
                          'bedrooms/0':'bedroom',
                          'pricingInfos/0/yearlyIptu':'tax',
                          'pricingInfos/0/price':'price',
                          'pricingInfos/0/monthlyCondoFee':'condominium'
                           },
                           inplace = True)


# #### There are many columns that we don't want to use, so we will select just the columns we want to work on

# In[8]:


df_vivareal = df_vivareal[['typeImovel','area','price','listingType','condominium','tax','neighborhood','city','state','bedroom','suites','bathroom','garage','latitude','longitude','address/street','address/streetNumber']]
df_vivareal


# In[9]:


df_vivareal.hist(bins=30, figsize = (30, 15))


# #### We don't want to work with multiple types of property, just apartment, so for that we select it.

# In[10]:


selecao_apartamentos = df_vivareal['typeImovel'] == "APARTMENT"
df_vivareal = df_vivareal[selecao_apartamentos]


# #### As we have many data, we want to visualize in the map how the data is distribuited in the city. 
# #### So, for that the first thing we did was to delet the data that contains NaN value in the latitude and longitude. Then, we wrote the code to create the map and then its settings

# In[11]:


df_vivareal = df_vivareal.dropna(subset=['latitude', 'longitude'])


# In[12]:


fig = px.scatter_mapbox(df_vivareal, lat="latitude", lon="longitude", color="price", size='price',
                        color_continuous_scale=px.colors.sequential.Turbo, size_max=35, opacity=0.5)


# In[13]:


fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_center_lon=df_vivareal['longitude'].mean(),
    mapbox_center_lat=df_vivareal['latitude'].mean(),
    mapbox_zoom=10,
    title='Interactive Map of Dispertion of Property Prices'
)


# # Analysing the data and making some graphics

# #### First of all, we want to visualize some descriptive statistics of the data

# In[14]:


df_vivareal.describe()


# #### Calcule for the m²

# In[15]:


df_vivareal['price_per_m2'] = df_vivareal['price'] / df_vivareal['area']


# ### Calcule for the mean price of the property of the neighborhood

# In[16]:


price_average_neighborhood = df_vivareal.groupby('neighborhood')['price'].mean().rename('price_average_neighborhood')


# ### Calcule for the m² of the neighborhood

# In[17]:


price_average_m2_neighborhood = df_vivareal.groupby('neighborhood')['price_per_m2'].mean().rename('price_average_m2_neighborhood')


# ### Adding theses values to the main dataframe

# In[18]:


df_vivareal = pd.merge(df_vivareal, price_average_neighborhood, how='left', on='neighborhood')
df_vivareal = pd.merge(df_vivareal, price_average_m2_neighborhood, how='left', on='neighborhood')


# ### This firts graphic shows the number of the property per neighborhood

# In[19]:


plt.figure(figsize=(20, 15))
sns.countplot(x='neighborhood', data=df_vivareal, palette='viridis',order=df_vivareal['neighborhood'].value_counts().index)
plt.title('Distribuition of property per neighborhood')
plt.xticks(rotation=45, ha='right')
plt.xlabel('neighborhood')
plt.ylabel('Nº of property')
plt.show()


# ### This graphic shows the mean price of each neighborhood

# In[20]:


plt.figure(figsize=(20, 10))
sns.barplot(x='neighborhood', y='price_average_neighborhood', data=df_vivareal, palette='viridis',
            order=df_vivareal.groupby('neighborhood')['price_average_neighborhood'].mean().sort_values(ascending=False).index)
plt.title('Average price of the neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Average Price (R$)')
plt.xticks(rotation=45, ha='right')
plt.show()


# ### In the real state, the price of the m² it's an information very important, and for that we made that graphic

# In[21]:


plt.figure(figsize=(20, 8))
df_vivareal['price_average_m2_neighborhood'] = df_vivareal['price'] / df_vivareal['area']
df_vivareal.groupby('neighborhood')['price_average_m2_neighborhood'].mean().sort_values().plot(kind='bar', color='orange')
plt.title('Average Price of the m² per Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('verage Price of the m² (R$)')
plt.xticks(rotation=45, ha='right')
plt.show()


# ### For visualize the correlation of the variables, we made a matrix of correlation. This graphic shows how one variables influence other. The correlation goes the 0 to 1, 0 been zero correlation and 1 a correlation very strong. 
# 
# #### We see that the variable that influence more the price is the quantity of suites and toilettes the apartament have

# In[22]:


correlation_matrix = df_vivareal[['price', 'area', 'condominium', 'tax', 'bedroom', 'suites', 'bathroom', 'garage']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrix of Correlation Between Variables')
plt.show()


# ## We choose to do a analysis of the 10 most expensive neighborhood of the city

# ### First we have to indentify the 10 most expensive neighborhood

# In[23]:


top10_bairros = df_vivareal.groupby('neighborhood')['price'].mean().nlargest(10).index


# ### We filtrated the dataframe to show just the 10 neighborhood

# In[24]:


df_top10 = df_vivareal[df_vivareal['neighborhood'].isin(top10_bairros)]


# ### And we want to visualize in the map where theses 10 neighborhood are in the map of the city

# In[25]:


fig = px.scatter_mapbox(df_top10, lat="latitude", lon="longitude", color="price", size='price',
                        color_continuous_scale=px.colors.sequential.Turbo, size_max=35, opacity=0.5)

# setting of the graphic
fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_center_lon=df_vivareal['longitude'].mean(),
    mapbox_center_lat=df_vivareal['latitude'].mean(),
    mapbox_zoom=11,
)
fig.show()


# ## For graphical analysis, we use different graphs to better understand the behavior of the variables.

# ### The boxplots show the price variation in the 10 most expensive neighborhoods. It can be seen whether there are large differences in price dispersion between these neighborhoods.

# In[26]:


plt.figure(figsize=(20, 10))
sns.boxplot(x='neighborhood', y='price', data=df_top10, order=top10_bairros)
plt.title('Price variation in the 10 most expensive neighborhoods')
plt.xlabel('Neighborhoods')
plt.ylabel('Price (R$)')
plt.xticks(rotation=45, ha='right')
plt.show()


# ### The scatter plot helps visualize the distribution of prices in relation to the area in the 10 most expensive neighborhoods. We can identify patterns or outliers that indicate properties with outliers.

# In[27]:


plt.figure(figsize=(15, 10))
sns.scatterplot(x='area', y='price', hue='neighborhood', data=df_top10, palette='viridis', size='price', sizes=(20, 400))
plt.title('Price spread in the 10 Most Expensive Neighborhoods')
plt.xlabel('Area (m²)')
plt.ylabel('Price (R$)')
plt.show()


# ### The swarm plot helps visualize the distribution of prices in relation to the price of m² in the 10 most expensive neighborhoods. We can better visualize the behavior of the price per m² in each neighborhood.

# In[28]:


plt.figure(figsize=(15, 8))
sns.swarmplot(data=df_top10, x = 'neighborhood', y = 'price_average_m2_neighborhood', palette="Dark2")


# ### The bar graph shows the average price per neighborhood in the 10 most expensive. This provides a clear view of the differences in average costs between the selected neighborhoods.

# In[29]:


plt.figure(figsize=(12, 8))
sns.barplot(x='neighborhood', y='price', data=df_top10, order=top10_bairros)
plt.title('Average Price per Neighborhood in the 10 Most Expensive')
plt.xlabel('Neighborhood')
plt.ylabel('Average Price (R$)')
plt.xticks(rotation=45, ha='right')
plt.show()


# ### The last bar graph shows the average cost per square meter in the 10 most expensive neighborhoods. This may indicate whether certain neighborhoods have a higher cost in relation to the area of the properties.

# In[30]:


plt.figure(figsize=(12, 8))
sns.barplot(x='neighborhood', y='price_average_m2_neighborhood', data=df_top10, order=top10_bairros)
plt.title('Cost per m² in the 10 Most Expensive Neighborhoods')
plt.xlabel('Neighborhoods')
plt.ylabel('Cost per m² (R$)')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[31]:


df_vivareal[df_vivareal['neighborhood'] == 'Petrópolis'].mean()


# In[32]:


df_vivareal[df_vivareal['address/street'] == 'Rua Faria Santos']


# # Prediction of Price

# In[33]:


df_cleaned = df_vivareal.drop(['typeImovel','listingType', 'address/street','city', 'state', 'address/streetNumber'], axis = 1)


# In[34]:


one_hot = pd.get_dummies(df_cleaned['neighborhood'])


# In[35]:


df_vivareal = df_cleaned.drop('neighborhood', axis = 1)
df_vivareal = df_vivareal.join(one_hot)


# In[36]:


df_vivareal = df_vivareal.dropna()


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


Y = df_vivareal['price']
X = df_vivareal.loc[:, df_vivareal.columns != 'price']


# In[39]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)


# In[40]:


from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()
rf_reg.fit(x_train, y_train)


# In[41]:


x_test


# In[42]:


n = 0
print(y_test.iloc[n])
print(x_test.iloc[n].head(10))


rf_reg.predict(x_test.iloc[n].values.reshape(1, -1))


# ## Conclusion
# ### By analyzing these graphs together, it is possible to identify patterns, trends and significant differences in real estate prices between neighborhoods. These findings can be helpful to buyers, sellers, and investors when making informed decisions about the real estate market.

# In[ ]:




