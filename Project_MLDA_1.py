#!/usr/bin/env python
# coding: utf-8

# In[79]:


get_ipython().system(' pip install pandas')
get_ipython().system(' pip install openpyxl')
get_ipython().system(' pip install seaborn')
get_ipython().system(' pip install matplotlib')
get_ipython().system(' pip install sklearn')


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# In[87]:


data1=pd.read_csv("E:\Desktop\MLDA-1\Project\DATA\Real estate valuation data set.csv")


# In[89]:


data1
data = data1.drop(['No'], axis = 1)


# In[90]:


data.head()


# In[91]:


data.describe()


# In[92]:


plt.figure(figsize=(16,6))
corr = data.corr()
sns.set_theme(style ='white')
heat_map = sns.heatmap(corr,vmin = -1, vmax = 1, square = True, annot = True, cmap = 'coolwarm')
heat_map.set_title('Correlation Heatmap', fontdict={'fontsize':18} )


# In[93]:


data.corr()


# In[94]:


data.count()


# In[95]:


sns.pairplot(data)


# In[96]:


data.boxplot(showmeans=True,vert= False,figsize=(25,10))


# In[97]:


data.columns


# In[99]:


# creating data set having required features
newdata = data.drop([ 'X1 transaction date', 'X5 latitude', 'X6 longitude'], axis = 1)


# In[48]:


newdata


# In[100]:


newdata.head()


# In[50]:


newdata.tail()


# In[51]:


newdata.corr()


# In[52]:


sns.pairplot(newdata)


# In[132]:


# Determining features and labels
# X refers to features and y refers to label
X = data.drop(['Y house price of unit area'], axis  = 1)
y = data['Y house price of unit area']


# In[134]:


#splitting dataset into two different partition for training/testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# In[133]:


y_test


# In[135]:


X_test


# In[136]:


X = data.iloc[:,:-1]
y = data.iloc[:,[-1]]


# In[137]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[138]:


X = scaler.fit_transform(x)
y = scaler.fit_transform(y)


# In[149]:


#fitting a model to training data
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[150]:


model.coef_


# In[152]:





# In[155]:


pd.DataFrame(model.coef_, X.columns, columns=['coeficients'])


# In[142]:


#Predicting price of unit area
y_pred = model.predict(X_test)


# In[145]:


# Evaluating the model
from sklearn import metrics
MAE = metrics.mean_absolute_error(y_test, y_pred) # MAE is short for Mean-Absolute-Error
MAE  


# In[144]:


MSE = metrics.mean_squared_error(y_test, y_pred)  # MSE is short for Mean-Square-Error
MSE


# In[146]:


RMSE = np.sqrt(MSE)  # RMSE is short for Root-Mean-Square-Error
RMSE


# In[ ]:





# In[130]:


test_residuals = y_test - y_pred


# In[91]:


sns.displot(data = test_residuals, bins = 30, kde = True)


# In[111]:


sns.regplot(x = y_pred ,y = y_pred)


# In[148]:


sns.regplot(x = y_pred ,y = y_test)


# In[ ]:





# In[ ]:




