#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[53]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm  import SVC


# # Reading Data

# In[3]:



data = pd.read_csv(r"C:\Users\Aryaa Ishika\Downloads\heart_failure.zip")


# In[4]:


data


# In[5]:


data.info()


# # Cleaning Data

# In[6]:


data.isnull().sum()


# In[7]:


data.describe(include='all')


# In[8]:


for i in data.columns:
    print(i,':',sum(data[i]=='?'))


# # Data Preprocessing

# In[9]:


X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state= 42)


# In[11]:


lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[12]:


y_pred = lr.predict(X_test)


# In[13]:


print(accuracy_score(y_test, y_pred))


# In[14]:


print(confusion_matrix(y_test, y_pred))


# In[15]:


sns.heatmap(confusion_matrix(y_test, y_pred), cmap='Reds', annot=True);


# In[17]:


print(classification_report(y_test, y_pred))


# In[19]:


# Distribution of Age

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Histogram(
    x = data['age'],
    xbins=dict( # bins used for histogram
        start=40,
        end=95,
        size=2
    ),
    marker_color='#e8ab60',
    opacity=1
))

fig.update_layout(
    title_text='AGE DISTRIBUTION',
    xaxis_title_text='AGE',
    yaxis_title_text='COUNT', 
    bargap=0.05, # gap between bars of adjacent location coordinates
    xaxis =  {'showgrid': False },
    yaxis = {'showgrid': False },
    template = 'plotly_dark'
)

fig.show()


# In[22]:


#Death events
import plotly.express as px
fig = px.histogram(data, x="age", color="DEATH_EVENT", marginal="violin", hover_data=data.columns, 
                   title ="Distribution of AGE Vs DEATH_EVENT", 
                   labels={"age": "AGE"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                  )
fig.show()


# In[23]:


# Creatinine Phosphokinase vs Death Event

import plotly.express as px
fig = px.histogram(data, x="creatinine_phosphokinase", color="DEATH_EVENT", marginal="violin", hover_data=data.columns,
                   title ="Distribution of CREATININE PHOSPHOKINASE Vs DEATH_EVENT", 
                   labels={"creatinine_phosphokinase": "CREATININE PHOSPHOKINASE"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# In[24]:


#Ejection vs Death Event
import plotly.express as px
fig = px.histogram(data, x="ejection_fraction", color="DEATH_EVENT", marginal="violin", hover_data=data.columns,
                   title ="Distribution of EJECTION FRACTION Vs DEATH_EVENT", 
                   labels={"ejection_fraction": "EJECTION FRACTION"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# In[26]:


# "Age Vs Diabetes"
import plotly.express as px
fig = px.histogram(data, x="age", color="diabetes", marginal="violin",hover_data=data.columns,
                   title ="Distribution of AGE Vs DIABETES", 
                   labels={"diabetes": "DIABETES", "age": "AGE"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# In[30]:


plt.style.use("seaborn")
for column in data.columns:
    if data[column].dtype!="object":
        plt.figure(figsize=(15,6))
        plt.subplot(2,2,1)
        sns.histplot(data=data,x=column,kde=True)
        plt.ylabel("freq")
        plt.xlabel(column)
        plt.title(f"distribution of {column}")
        plt.subplot(2,2,2)
        sns.boxplot(data=data,x=column)
        plt.ylabel(column)
        plt.title(f"boxplot of {column}")
        plt.show()


# # Logistic Regression and Splitting Data

# In[33]:


x = data.drop("DEATH_EVENT", axis=1)
y = data['DEATH_EVENT']


# In[34]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x = scaler.fit_transform(x)
y = scaler.fit_transform(y.values.reshape(-1,1))


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42, stratify = y)

print(f"The Shape of x_train : {x_train.shape}")
print(f"The Shape of x_test : {x_test.shape}")
print(f"The Shape of y_train : {y_train.shape}")
print(f"The Shape of y_test : {y_test.shape}")


# In[36]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[37]:


y_pred = model.predict(x_test)


# In[38]:


print(classification_report(y_test,y_pred))


# # Decision Tree

# In[41]:


classifier = DecisionTreeClassifier(max_leaf_nodes = 3, random_state=0, criterion='entropy')
classifier.fit(x_train, y_train)


# In[43]:


y_predd = classifier.predict(x_test)
y_predd


# In[45]:


print(classification_report(y_test,y_pred))


# # Support Vector Classifier (SVM)
# 

# In[48]:


svm = SVC(C = 0.6, random_state = 42, kernel='rbf')
svm.fit(x_train, y_train)


# In[49]:


y_pred = svm.predict(x_test)
print(y_pred)


# In[51]:


print(classification_report(y_test,y_pred))


# # K Nearest Neighbor

# In[54]:


kn = KNeighborsClassifier(n_neighbors=6)
kn.fit(x_train, y_train)


# In[55]:


y_pred = kn.predict(x_test)
print(y_pred)


# In[56]:


print(classification_report(y_test,y_pred))


# # Random Forest Classifcation

# In[57]:


classifier = RandomForestClassifier(n_estimators=11, criterion='entropy', random_state=42)
classifier.fit(x_train, y_train)


# In[58]:


y_pred = classifier.predict(x_test)
print(y_pred)


# In[59]:


print(classification_report(y_test,y_pred))


# # Model Evaluation

# In[ ]:




