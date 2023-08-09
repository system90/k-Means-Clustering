#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation - Using K-means Clustering

# Looking at customer dataset segmentations using unsupervised machine learning algorithm k-means clustering.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans


# In[3]:


data = pd.read_csv('Mall_Customers.csv')


# In[4]:


# check shape of dataframe
data.shape


# In[5]:


# view first rows of dataset
data.head()


# In[6]:


# view statistics of dataset
data.describe()


# In[7]:


data.info()


# The output above shows that 'Gender' is the only variable in the dataset that is non integer.

# In[8]:


# create a new dataframe, using only 'Annual income' and 'Spending score'
# use iloc to return selected columns and rows

data2 = data.iloc[:, [3,4]].values      # using columns 3 and 4


# In[9]:


# Check the values of new dataset
data2


# ### Establish the optimum number of clusters

# In[19]:


# create a blank list
wcss = []                           # wcss: Within Cluster Sum of Squares


# In[20]:


# train model
for i in range(1,15):                                                       # run up to a maximum of 15 clusters
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)        # using a smart initialiser
    kmeans.fit(data2)
    wcss.append(kmeans.inertia_)


# In[22]:


plt.figure(figsize = (8,6))                              # Plotsize
plt.plot(range(1,15), wcss)
plt.title('Elbow Method', fontsize=22)  
plt.xlabel('Annual Income (k$)')
plt.ylabel('WCSS Values')
plt.show()


# The above elbow plot is used to establish the optimal k value for the dataset. From visual inspection the point on the line with the largest angle occurs at 5 on the x-axis (this is where the rate of descent slows). Therefore the optimal k value for this dataset is 5.

# In[13]:


# Initialise the k-means model

kmeansmodel = KMeans(n_clusters = 5, init='k-means++', random_state=0)     # init: method for initilisation 


# In[14]:


# Use model to make predictions

pred_kmeans = kmeansmodel.fit_predict(data2)


# In[18]:


# Plot 
plt.figure(figsize = (10,8))                             
plt.title('Customer Cluster Analysis', fontsize=22)
plt.xlabel('Annual Income ($k)')
plt.ylabel('Spending Score (1-100)')

# Plot the 5 clusters
# each clusters is assigned a point in the matrix
plt.scatter(data2[pred_kmeans==0,0], data2[pred_kmeans==0,1], s=60, c="orange", label='Customer 1')     # 
plt.scatter(data2[pred_kmeans==1,0], data2[pred_kmeans==1,1], s=60, c="navy", label='Customer 2')   
plt.scatter(data2[pred_kmeans==2,0], data2[pred_kmeans==2,1], s=60, c="green", label='Customer 3') 
plt.scatter(data2[pred_kmeans==3,0], data2[pred_kmeans==3,1], s=60, c="purple", label='Customer 4') 
plt.scatter(data2[pred_kmeans==4,0], data2[pred_kmeans==4,1], s=60, c="olive", label='Customer 5')

#Plot the centroids:
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60, c ='red', label = 'Centroids')


plt.legend()
plt.show()


# Upon visual inspection we can see that cluster 'Customer 2' contains the most amount of points than the other clusters.
