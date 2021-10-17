#!/usr/bin/env python
# coding: utf-8

# In[14]:


#Importing the required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


# Importing the dataset we need
data1=pd.read_csv("E:\Desktop\MLDA-1\Project\DATA\Real estate valuation data set.csv")
data = data1.drop(['No'], axis = 1)
data


# In[16]:


#Checking dataset for the NaN values and other EDAs
data.head(5)


# In[17]:


data.describe()


# In[18]:


#Plotting the realtions
sns.regplot(x = "X1 transaction date" ,y = "Y house price of unit area",data = data)
#You can see the relation between the year and the price.


# In[19]:


sns.regplot(x = "X2 house age" ,y = "Y house price of unit area",data = data)


# In[ ]:


#You can see that the more the house age less the price in most case


# In[20]:


sns.regplot(x = "X3 distance to the nearest MRT station" ,y = "Y house price of unit area",data = data)


# In[ ]:


#You can see if the house is near the MRT station then the prices is high as compare to far places


# In[21]:


sns.regplot(x = "X4 number of convenience stores" ,y = "Y house price of unit area",data = data)


# In[ ]:


# If more stores near the house the prices is high


# In[22]:


sns.regplot(x = "X5 latitude" ,y = "Y house price of unit area",data = data)


# In[23]:


sns.regplot(x = "X6 longitude" ,y = "Y house price of unit area",data = data)


# In[24]:


plt.figure(figsize=(8,3))
sns.displot(x=data['Y house price of unit area'], kde=True, aspect=2, color='purple')
plt.xlabel('house price of unit area')


# In[25]:


data.isnull().sum()


# In[26]:


x = data.iloc[:,:-1]
y = data.iloc[:,[-1]]


# In[19]:


# Before fitting the data to the model let's normalise tha data


# In[27]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[28]:


x = scaler.fit_transform(x)
y = scaler.fit_transform(y)


# In[30]:


# Spliting of the dataset
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20, random_state=101)


# In[31]:


#  regression problem
#  Ridge Regression


# In[32]:


from sklearn.linear_model import Ridge

reg = Ridge(alpha = 30)


# In[33]:


reg.fit(x_train,y_train)


# In[34]:


pred = reg.predict(x_test)


# In[35]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

MAE= mean_absolute_error(y_test, pred)
MSE= mean_squared_error(y_test, pred)
RMSE= np.sqrt(MSE)


# In[36]:


Error = {"Mean_Absolute_Error" : MAE,"Mean_Square_Error": MSE,"Root_Mean_Square_Error":RMSE}


# In[69]:


Error


# In[37]:


sns.regplot(x = pred ,y = pred)


# In[38]:


sns.regplot(x = pred ,y = y_test)


# In[ ]:




