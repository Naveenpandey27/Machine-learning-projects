#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import svm
from sklearn import datasets


# In[2]:


iris = datasets.load_iris()


# In[3]:


type(iris)


# In[39]:


print(iris.data)


# In[5]:


iris['target']


# In[6]:


iris.target_names


# In[7]:


x = iris.data[:,2]
y = iris.target


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# In[9]:


model = svm.SVC(kernel = 'linear')


# In[10]:


model.fit(x_train, y_train)


# In[11]:


y_pred = model.predict(x_test)


# In[12]:


y_test


# In[13]:


from sklearn.metrics import accuracy_score


# In[14]:


print(accuracy_score(y_test, y_pred))


# In[15]:


from sklearn.metrics import confusion_matrix


# In[16]:


cm = confusion_matrix(y_pred, y_test)
cm


# # KNN

# In[17]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[18]:


x = iris.data
y = iris.target


# In[19]:


x.shape


# In[20]:


y.shape


# In[21]:


from sklearn.neighbors import KNeighborsClassifier


# In[22]:


knn = KNeighborsClassifier(n_neighbors = 1)


# In[23]:


knn


# In[24]:


knn.fit(x, y)


# In[25]:


import numpy as np
a = np.array([4,5,4,6])
a.shape


# In[26]:


pred1 = knn.predict([a])
pred1


# In[27]:


iris['target_names']


# In[28]:


from sklearn.linear_model import LogisticRegression


# In[29]:


lr = LogisticRegression()


# In[30]:


lr.fit(x_train, y_train)


# In[31]:


y_pred = lr.predict(x_test)


# In[32]:


y_pred


# In[33]:


y_test


# In[34]:


from sklearn.metrics import confusion_matrix


# In[35]:


cm = confusion_matrix(y_test, y_pred)


# In[36]:


cm


# In[37]:


from sklearn.metrics import accuracy_score


# In[38]:


print(accuracy_score(y_test, y_pred))


# In[ ]:




