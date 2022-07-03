#!/usr/bin/env python
# coding: utf-8

# ## IMANE BELBACHIR
# 

# DATA SCIENCE & BUSINESS ANALYTICS INTERN AT THE SPARK FOUNDATION
# 

# TSF GRIP TASK

# prediction using supervised ML

# # dataset: http://bit.ly/w-data

# In[179]:


#importing librairies
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # READING DATA

# In[180]:


DF=pd.read_csv("http://bit.ly/w-data")
#we will use DF.shape so we can know how much lines and rows we have
print(DF.shape)


# In[181]:


DF.head(26)


# # distribution of scores

# In[182]:


DF.plot(x='Hours',y='Scores',style='1')
plt.title("hours vs pourcentage")
plt.xlabel("hours studies")
plt.ylabel("pourcentages scores")
plt.show()


# # preparation of the data

# In[183]:


x=DF.iloc[:,:-1].values
y=DF.iloc[:, 1].values


# # training and test sets
# 

# In[184]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# # simple linear regression

# In[185]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print("trained algorithm")


# # Regression line for the test data
# 

# In[186]:



print(regressor.coef_)


# In[187]:


print( regressor.intercept_)


# In[188]:


line=regressor.coef_*x+regressor.intercept_
print(line)


# In[189]:


print(y.shape)


# In[190]:


plt.scatter(x,y,color="red")
plt.plot(x,line,color="blue")
plt.show()


# # prediction
# 

# In[191]:


y_prediction=regressor.predict(x_test)
print(y_prediction)


# In[192]:


dataframe= pd.DataFrame({'A':y_test,'prediction':y_prediction})
print(dataframe)


# In[195]:


hours=np.array([[9.5]])
p=regressor.predict(hours)
print("number of hours={}".format(hours[0][0]))
print("prediction scores={}".format(p[0]))


# In[196]:


from sklearn import metrics
print("mean absolute error",metrics.mean_absolute_error(y_test,y_prediction))


# In[ ]:




