#!/usr/bin/env python
# coding: utf-8

# In[86]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori
import utils
get_ipython().run_line_magic('matplotlib', 'inline')


# In[87]:


trip_review=pd.read_csv('tripadvisor_review.csv')


# In[88]:


trip_review


# In[89]:


# rename columns names
trip_review.columns = ['ID','Galleries', 'Clubs', 'Bars', 'Restaurants',
                       'Museums','Resorts','Parks','Beaches','Theaters','Institutions']


# In[90]:


trip_review.columns


# In[91]:


trip_review.head()


# In[92]:


trip_review.tail()


# In[93]:


# check ID is it unique
len(trip_review) == len(trip_review['ID'].unique())


# In[94]:


trip_review = trip_review.set_index(['ID'])


# In[95]:


trip_review.head()


# In[96]:


trip_review.info()


# In[97]:


trip_review.describe()


# In[98]:


#changing the data to make it suitable 
trip_review['Galleries']= np.where(trip_review['Galleries']>=0.89,'Yes','No')


# In[99]:


trip_review['Clubs']= np.where(trip_review['Clubs']>=1.35,'Yes','No')


# In[100]:


trip_review['Bars']= np.where(trip_review['Bars']>=1.01,'Yes','No')


# In[101]:


trip_review['Restaurants']= np.where(trip_review['Restaurants']>=0.53,'Yes','No')


# In[102]:


trip_review['Museums']= np.where(trip_review['Museums']>=0.93,'Yes','No')


# In[103]:


trip_review['Resorts']= np.where(trip_review['Resorts']>=1.84,'Yes','No')


# In[104]:


trip_review['Parks']= np.where(trip_review['Parks']>=3.18,'Yes','No')


# In[105]:


trip_review['Beaches']= np.where(trip_review['Beaches']>=2.83,'Yes','No')


# In[106]:


trip_review['Theaters']= np.where(trip_review['Theaters']>=1.56,'Yes','No')


# In[107]:


trip_review['Institutions']= np.where(trip_review['Institutions']>=2.79,'Yes','No')


# In[108]:


trip_review.head()


# In[109]:


trip_review.tail()


# In[110]:


trip_review.shape


# In[111]:


trip_review.size


# In[112]:


# check the data have missing values
count= trip_review.isnull().sum().sort_values(ascending=False)
percentage = ((trip_review.isnull().sum()/len(trip_review))*100).sort_values(ascending=False)
missing_data = pd.concat([count,percentage],axis=1,keys=['Count','Percentage'])
print ('Count and Percentage of missing values for the columns:')
missing_data


# In[113]:


yes=(trip_review=='Yes').sum()
no=(trip_review=='No').sum()
Review=pd.concat([yes,no],axis=1,keys=['yes','no'])
ax=Review.plot.bar(stacked=True)


# In[114]:


transactions=utils.data_prepare(trip_review)
transactions


# In[115]:


Rules=list(apriori(transactions,min_support=0.05,min_confidence=0.8))
associationRules=utils.extract(Rules)
rules_trip_review=pd.DataFrame(associationRules,columns=['LHS','RHS','Support','Confidence','Lift'])
len(rules_trip_review)


# In[116]:


rules_trip_review.nlargest(15,"Lift")


# In[117]:


rules_trip_review.nlargest(15,"Support")


# In[118]:


rules_trip_review.nlargest(15,"Confidence")


# In[119]:


rules_trip_review[rules_trip_review['LHS'].apply(lambda x:len(x)>0)].nlargest(10,'Support')


# In[120]:


Rules=list(apriori(transactions,min_support=0.05,min_confidence=0.8,max_length=3))
associationRules=utils.extract(Rules)
rules_trip_review=pd.DataFrame(associationRules,columns=['LHS','RHS','Support','Confidence','Lift'])
len(rules_trip_review)


# In[121]:


rules_trip_review.nlargest(15,'Lift')


# In[122]:


rules_trip_review.nlargest(15,'Support')


# In[123]:


rules_trip_review.nlargest(15,'Confidence')


# In[124]:


ax=Review.plot.bar()


# In[125]:


Rules=list(apriori(transactions,min_support=0.5,min_confidence=0.9))
associationRules=utils.extract(Rules,'Parks',2)
utils.inspect(associationRules)


# In[126]:


Rules=list(apriori(transactions,min_support=0.1,min_confidence=0.9))
associationRules=utils.extract(Rules,'Parks',2)
utils.inspect(associationRules)


# In[127]:


Rules=list(apriori(transactions,min_support=0.2,min_confidence=0.9))
associationRules=utils.extract(Rules,'Parks',2)
utils.inspect(associationRules)


# In[129]:


rules_trip_review = pd.DataFrame(associationRules,columns=['LHS','RHS','Support','Confidence','Lift'])
import plotly.express as px
fig = px.scatter(rules_trip_review, x='Support',y='Confidence',color='Lift',
                hover_data=['LHS','RHS'],color_continuous_scale='agsunset')
fig.show() 


# In[130]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.figure(figsize=(10,8))
ax = plt.axes(projection = '3d')
fig = ax.scatter3D(rules_trip_review['Support'],rules_trip_review['Confidence'],rules_trip_review['Lift'],alpha=1,lw=3,
                  c = rules_trip_review['Lift'])
ax.set_xlabel('Support')
ax.set_ylabel('Confidence')
ax.set_zlabel('Lift')
plt.colorbar(fig)

