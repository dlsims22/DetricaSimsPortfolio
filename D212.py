#!/usr/bin/env python
# coding: utf-8

# # #Import Required Packages and Libraries

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from mlxtend.preprocessing import TransactionEncoder 
from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules


# # Import Dataset

# In[2]:


#Import Dataset
Telco= pd.read_csv ('D:/telco_market_basket.csv')


# # Get information on dataset

# In[3]:


Telco.info()


# In[4]:


#Get dataset information
Telco.head()


# # Check for missing data

# In[5]:


#Remove blank columns
Telco.isnull().sum


# In[6]:


#Check for missing values
Telco.isnull()


# # Remove blank rows 

# In[7]:


Telco.dropna(axis=0, how = 'all', thresh= None, subset = None, inplace=True)


# # Review dataset dimensions

# In[8]:


#Review Dataset Dimensions
Telco.shape


# In[9]:


# Review datatypes
print(Telco.dtypes)


# # Create list of lists to prepare dataset for encoding 

# In[10]:


trans = []
for i in range (0, 7501):
    trans.append([str(Telco.values[i,j]) for j in range (0, 20)])


# # Transactionalize data to prepare for Apriori function

# In[11]:


TE = TransactionEncoder()
array = TE.fit(trans).transform(trans)


# # Convert dataset back to dataframe

# In[12]:


cleaned_Telco = pd.DataFrame(array, columns = TE.columns_)
cleaned_Telco


# # List items as columns

# In[13]:


for col in cleaned_Telco.columns:
    print(col)


# # Drop empty columns

# In[14]:


Telco_cleaned = cleaned_Telco.drop(['nan'], axis =1)


# In[15]:


for col in Telco_cleaned.columns:
       print(col)


# In[16]:


Telco_cleaned.shape


# # Export cleaned Dataset

# In[25]:


import pandas as pd
Telco_cleaned.to_csv("D:/Cleaned_Telco_MBA.csv")


# # Run Apriori function to mine rules and review head info

# In[18]:


a_rules = apriori(Telco_cleaned, min_support = 0.05, use_colnames = True)
a_rules.head()


# # Set metrics and mine rules

# In[19]:


ass_r = association_rules(a_rules, metric = 'lift', min_threshold = 1)
ass_r.head(3)


# # Recommend items based on rules

# In[20]:


Telco_cleaned['HP 61 ink'].sum()


# In[21]:


Telco_cleaned['Dust-Off Compressed Gas 2 pack'].sum()


# # Determine rules based on set conditions

# In[22]:


ass_r [ (ass_r['lift'] >= 1.40) &
        (ass_r['confidence'] >=0.33) ]
        


# In[23]:


Telco_cleaned['VIVO Dual LCD Monitor Desk mount'].sum()


# In[24]:


Telco_cleaned['Dust-Off Compressed Gas 2 pack'].sum()

