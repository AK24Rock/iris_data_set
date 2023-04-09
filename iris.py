#!/usr/bin/env python
# coding: utf-8

# # 

# # Import modules

# In[80]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the dataset
# 

# In[142]:


df = pd.read_csv('Iris.csv')
df.head()


# In[143]:


#To delet a column
df = df.drop(columns = ['Id'])
df


# In[83]:


#Todescribe stat about data
df.describe()


# In[84]:


#Basic info about data type
df.info()


# In[85]:


#To display no. of samples on each class
df['Species'].value_counts()


# # Preprocessing the dataset

# In[86]:


#check for null values
df.isnull().sum()


# In[ ]:





# # Exploratory Data Analysis

# In[87]:


df['SepalLengthCm'].hist()


# In[88]:


df['SepalWidthCm'].hist()


# In[89]:


df['PetalLengthCm'].hist()


# In[90]:


df['PetalWidthCm'].hist()


# In[91]:


# Scatter plot
sns.scatterplot(x='SepalWidthCm', y='PetalLengthCm', data = df, hue ='Species')


# In[92]:


g = sns.pairplot(df, hue='Species', markers='+')
plt.show()


# # Coorelation Matrix

# In[93]:


df.corr()


# In[94]:


#display the cprrelation we use heat map
corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))#size
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')# cmap = 'coolwarm' for colour on the basis of corr


# In[95]:


# here PetalWidth and PetalLength have higher corr. close to the 1 we can ignor any one from the both
# red show higher corr and blue shows lower


# # Label Encoder

# In[96]:


# To convert all the categories into the int form because some time it is string for that we usew label encoder
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder() # initalise the function


# In[97]:


#df['Species'] = le.fit_transform(df['Species'])# here we use categorial coloum always
#df.head()


# In[98]:


# String variable is converted to the int 0,1,2


# # Model Traning

# In[144]:


from sklearn.model_selection import train_test_split
# train - 70%
# test - 30%
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30)


# In[145]:


#Logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()# define the model with LRL


# In[146]:


model.fit(x_train, y_train)


# In[147]:


#print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)# *100 for percentage


# In[148]:


# Knn
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[149]:


model.fit(x_train, y_train)


# In[150]:


#print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)# *100 for percentage


# In[151]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[152]:


model.fit(x_train, y_train)


# In[153]:


#print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)# *100 for percentage


# In[154]:


#Save the model 
import pickle
filename = 'savedmodel.sav'
pickle.dump(model, open(filename, 'wb'))


# In[155]:


load_model = pickle.load(open(filename, 'rb'))


# In[156]:


x_test.head()


# In[157]:


load_model.predict([[5.0, 3.5, 1.3, 0.3]])


# In[ ]:




