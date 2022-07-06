#!/usr/bin/env python
# coding: utf-8

# # Importing Packages

# In[ ]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# # Import Dataset

# In[ ]:


dataset = pd.read_csv(sys.argv[1])


# Plot Data

# In[ ]:


plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# Fitting Linear Regression to the Dataset

# In[ ]:


model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])


# Adjusted R-Squared

# In[ ]:


model.score(dataset[['x']], dataset[['y']])


# # Visualizing the Linear Regression results

# In[ ]:


plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

