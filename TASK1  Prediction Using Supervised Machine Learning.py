#!/usr/bin/env python
# coding: utf-8

# # SURABHI DHADIWAL 

# # Task-1 : Prediction Using Supervised Machine Learning 
# **Objective**:To predict percentage score of student who studies for 9.25 hours in a day.
# 

# **Importing libraries and dataset**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats


# In[2]:


data_import="http://bit.ly/w-data"
student_data=pd.read_csv(data_import)
student_data.head()


# In[3]:


df=pd.DataFrame(student_data)
print(df.describe())


# **Data Visualization**

# In[4]:


sn.distplot(student_data['Scores'],color='Pink')


# In[5]:


sn.distplot(student_data['Hours'])


# In[7]:


corrmatrix=df.corr()
print(corrmatrix)
sn.heatmap(corrmatrix,annot=True,cmap='coolwarm')
plt.show()


# In[8]:


student_data.plot(x='Hours',y='Scores',style='o')
plt.title('Scatter plot')
plt.xlabel('Study Hours')
plt.ylabel('Percentage Scored')
plt.show()


# **Train Test Split**

# In[9]:


#splitting data into 80% train and 20% test
x= student_data.iloc[:,:-1].values
print(x)
y= student_data.iloc[:,1].values
print(y)
X_train, X_test, Y_train, Y_test=train_test_split(x, y,  test_size=0.2, random_state=0)


# **Model fitting**

# In[10]:


regressor=LinearRegression()
regressor.fit(X_train,Y_train)
print("Training complete successfully")


# **Plotting Regression Line**

# In[11]:


line=regressor.coef_*x+regressor.intercept_
plt.scatter(x,y,color='red')
plt.plot(x,line)
plt.show()


# In[12]:


print(X_test)
y_pred=regressor.predict(X_test)
print(y_pred)


# **Actual vs Predicted**

# In[13]:


df=pd.DataFrame({'Actual':Y_test,'Predicted':y_pred})
df


# In[14]:


df.plot(kind='bar')
plt.xlabel('Hours')
plt.ylabel('Scores')


# **Predicted score when student studies for 9.25 hours in a day**

# In[15]:


hours = 9.25
own_pred=regressor.predict([[hours]])
print("No of Hours=",format(hours))
print("Predicted score=",format(own_pred[0]))


# **Evaluating Model Accuracy**

# In[16]:


print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test,y_pred))


# In[17]:


slope,intercept,r,p,std_err=stats.linregress(Y_test,y_pred)
print('R-squared Value:',r)


# **ThankYou!**
