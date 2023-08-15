#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
teams = pd.read_csv('teams.csv')


# In[35]:


teams


# In[36]:


teams = teams[["team","country","athletes","year","events","age","prev_medals","medals"]]
teams


# In[39]:


teams.corr()["medals"]


# In[40]:


import seaborn as sns


# In[41]:


sns.lmplot(x='athletes',y='medals',data=teams,fit_reg=True,ci=None)


# In[42]:


sns.lmplot(x='age',y='medals',data=teams,fit_reg=True,ci=None)


# 

# In[43]:


teams.plot.hist(y='medals')


# In[44]:


teams[teams.isnull().any(axis=1)]


# In[45]:


teams = teams.dropna()


# In[46]:


teams


# In[49]:


train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()


# In[51]:


from sklearn.linear_model import LinearRegression


# In[52]:


reg = LinearRegression()


# In[53]:


predictors = ["athletes", "prev_medals"]
target = ["medals"]
reg.fit(train[predictors],train[target])


# In[55]:


predictions = reg.predict(test[predictors])


# In[56]:


predictions


# In[57]:


test["predictions"] = predictions


# In[67]:


test


# In[63]:


test.loc[test["predictions"] < 0 , "predictions"] = 0


# In[64]:


test


# In[65]:


test["prediction"] = test["predictions"].round()


# In[66]:


test


# In[ ]:




