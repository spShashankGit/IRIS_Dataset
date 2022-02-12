#!/usr/bin/env python
# coding: utf-8

# # IRIS

# ##### Import libraries

# In[2]:


import pandas as pd
import numpy as np


# ##### Read dataset

# In[3]:


df = pd.read_csv('Dataset/iris.data', index_col=0, header=None)


# ##### Perform 1 hot encoding on the dataset

# In[4]:


df


# In[5]:


df = pd.get_dummies(df)


# In[6]:


df


# In[7]:


df = pd.get_dummies(df)


# ##### Shuffle the whole dataset
# Shuffling the dataset before splitting because if we shuffle the dataset manually after splitting then the problem of mis-match data and label will occur.

# In[27]:


df = df.sample(frac=1)


# In[51]:


lenghtofData = len(df.index)
lenghtofData


# In[63]:


df.iloc[:3,:]


# In[70]:


df.iloc[2:6,:]


# In[81]:


lenghtofData = len(df.index)

train_data   = df.iloc[0:int((lenghtofData*75)/100),:]
train_data


# ##### Split dataset into train and test dataset

# 1. Idea is to make a generic function to split the train and test dataset. 
# 2. The ratio of train:test is a subject of discussion. 
# 3. To improve that discussion I have an idea. 
# 4. The idea is to train our network with different rations and record the observation. 
# 5. As a step one we can take 70:30 or please suggest if some paper have a better idea. 
# 6. After training the model once we could try with 75:25; 80:20 etc. depending upon our bandwidth and the motivation ;)

# In[94]:


def split_train_test(data,tr, tst):
    
    lenghtofData = len(data.index)
    
    train_data_comp   = data.iloc[0:int((lenghtofData*tr)/100),:]
    test_data_comp    = data.iloc[0:int((lenghtofData*tst)/100),:]
    
    train_data    = train_data_comp[train_data_comp.columns[0:3]]
    train_label   = train_data_comp[train_data_comp.columns[3:]]
    
    test_data    = test_data_comp[test_data_comp.columns[0:3]]
    test_label   = test_data_comp[test_data_comp.columns[3:]]
    return train_data,train_label,test_data, test_label


# In[95]:


trainData, trainLabel, testData, testLabel = split_train_test(df, 75,25)


# In[ ]:




