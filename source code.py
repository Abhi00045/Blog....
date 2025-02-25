#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
# Ignore all warnings
warnings.filterwarnings('ignore')
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,SVR
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv(r"Malaria.csv")
df


# In[3]:


from sklearn.utils import resample

df = resample(df, replace=True, n_samples=39000, random_state=42)


# In[4]:


df.head()


# In[5]:


y=df['outbreak_threshold']


# In[6]:


import pandas as pd

# Check the data types of each column
print(df.dtypes)


# In[7]:


print(df.dtypes)


# In[8]:


type(df)


# In[9]:


df['mosquito_species'].unique()


# In[10]:


## null 

df.isnull().sum()


# In[11]:


df.head(2)


# ## LabelEncoder

# In[12]:


# Initialize the LabelEncoder
le= LabelEncoder()

# Loop through each column in the DataFrame
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])
df.head()


# In[13]:


df.info()


# In[14]:


X = df.drop(columns=['outbreak_threshold'])
y=df['outbreak_threshold']


# In[15]:


y


# In[16]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)


# In[17]:


X_train


# In[18]:


X_test


# In[19]:


y_train


# In[20]:


y_test


# In[21]:


df.head(2)


# In[22]:


plt.figure(figsize = (25,10))
ax = sns.barplot(x="mosquito_species", y= "outbreak_threshold", data=df[:18] ,palette = "Spectral")
plt.title (" Number outbreak_threshold")
plt.xticks(rotation = 60, ha = 'right')
plt.xlabel("mosquito_species")
plt.ylabel("outbreak_threshold")

plt.show()


# In[23]:


df['mosquito_species']


# In[24]:


#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    testY = testY.astype('int')
    predict = predict.astype('int')
    mse = mean_squared_error(testY, predict)
    mae = mean_absolute_error(testY, predict)
    r2 = r2_score(testY, predict) * 100
    
    mean_squared_error.append(mse)
    mean_absolute_error.append(mae)
    r2_score.append(r2)
    
    print(algorithm+' Accuracy    : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' FSCORE      : '+str(f))
    report=classification_report(predict, testY,target_names=labels)
    print('\n',algorithm+" classification report\n",report)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()


# ## Decision Tree

# In[25]:


import os
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Example function to calculate metrics (you can modify it as needed)
def calculate_metrics(model_name, y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Metrics:")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

# Check if the model file exists
if os.path.exists('Decision_tree_regressor_model.pkl'):
    # Load the trained model from the file
    reg = joblib.load('Decision_tree_regressor_model.pkl')
    print("Model loaded successfully.")
    y_pred = reg.predict(X_test)
    calculate_metrics("DecisionTreeRegressor", y_pred, y_test)
else:
    # Train the model (assuming X_train and y_train are defined)
    reg = DecisionTreeRegressor(max_depth=10, random_state=0)
    reg.fit(X_train, y_train)
    # Save the trained model to a file
    joblib.dump(reg, 'Decision_tree_regressor_model.pkl')
    print("Model saved successfully.")
    y_pred = reg.predict(X_test)
    calculate_metrics("DecisionTreeRegressor", y_pred, y_test)


# ## MLP

# In[26]:


import os
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Example function to calculate metrics (you can modify it as needed)
def calculate_metrics(model_name, y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Metrics:")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

# Check if the model file exists
if os.path.exists('MLP_regressor_model.pkl'):
    # Load the trained model from the file
    reg = joblib.load('MLP_regressor_model.pkl')
    print("Model loaded successfully.")
    y_pred = reg.predict(X_test)
    calculate_metrics("MLPRegressor", y_pred, y_test)
else:
    # Train the model (assuming X_train and y_train are defined)
    reg = MLPRegressor(hidden_layer_sizes=(180, 180), max_iter=1000, random_state=0)
    reg.fit(X_train, y_train)
    # Save the trained model to a file
    joblib.dump(reg, 'MLP_regressor_model.pkl')
    print("Model saved successfully.")
    y_pred = reg.predict(X_test)
    calculate_metrics("MLPRegressor", y_pred, y_test)


# ## test data

# In[27]:


test=pd.read_csv(r'test.csv')
test.head()


# In[28]:


# Initialize the LabelEncoder
le= LabelEncoder()

# Loop through each column in the DataFrame
for column in test.columns:
    if test[column].dtype == 'object':
        test[column] = le.fit_transform(test[column])
df.head()


# In[29]:


X_test.shape


# In[30]:


test.shape


# In[31]:


# Make predictions on the selected test data
predict = reg.predict(test)
predict


# In[32]:


predict


# In[33]:


predict=pd.DataFrame(predict)
predict.head()


# In[34]:


test=pd.read_csv(r'test.csv')
test.head()


# In[35]:


test['predict']=predict
test.head()


# In[ ]:




