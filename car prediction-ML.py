#!/usr/bin/env python
# coding: utf-8

# IMPORTING DEPENDENCIES

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from datetime import datetime


# DATA COLLECTION AND PROCESSING

# In[13]:


car_data_1=pd.read_csv(r"C:\Users\sujil\OneDrive\Documents\Train.csv")


# In[14]:


car_data_1.head(10)


# In[15]:


car_data_1.set_index("id",inplace=True)


# In[16]:


car_data_1.head()


# In[17]:


car_data_1.isnull().sum()


# In[18]:


car_data_1.fillna(car_data.mean(), inplace=True)


# In[19]:


#car_data_1.dropna(inplace=True)


# In[20]:


car_data_1.shape


# In[21]:


car_data_1.duplicated().sum()


# In[22]:


car_data_1.drop(columns=["engine"],inplace=True)


# In[23]:


car_data_1.info()


# In[24]:


Q1 = car_data_1.quantile(0.25)
Q3 = car_data_1.quantile(0.75)

# Calculate the IQR
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers from the dataset
car_data = car_data_1[~((car_data_1 < lower_bound) | (car_data_1 > upper_bound)).any(axis=1)]


# In[25]:


car_data.shape


# #DATA ANALYSIS

# In[26]:


for col in car_data.columns:
    print("UNIQUE VALUE OF " + col)
    print(car_data[col].unique())
    print("##################################")


# In[27]:


print(car_data.brand.value_counts())
print(car_data.model_year.value_counts())
print(car_data.milage.value_counts())
print(car_data.fuel_type.value_counts())
print(car_data.transmission.value_counts())
print(car_data.ext_col.value_counts())
print(car_data.int_col.value_counts())
print(car_data.accident.value_counts())


# In[28]:


car_data.drop(columns=["clean_title"],inplace=True)


# In[29]:


current_year = datetime.now().year
car_data['car_age'] = current_year - car_data['model_year']
car_data.head()


# In[30]:


#car_data['horsepower_per_cylinder'] = car_data['hp'] / car_data['engine_cylinder'].replace(0, 1)  # Avoid division by zero
#car_data['horsepower_per_cylinder'] = car_data['horsepower_per_cylinder'].fillna(0)  # Set any resulting NaN to 0
#car_data.head()


# In[31]:


#car_data['mileage_per_year'] = np.where(car_data['car_age'] != 0, 
 #                                         car_data['milage'] / car_data['car_age'], 
  #                                        0) 


# In[32]:


#print(car_data.model.value_counts())


# In[33]:


car_data['model'] = car_data.groupby('model')['price'].transform('mean')
car_data['transmission'] = car_data.groupby('transmission')['price'].transform('mean')
car_data['ext_col'] = car_data.groupby('ext_col')['price'].transform('mean')
car_data['int_col'] = car_data.groupby('int_col')['price'].transform('mean')




# In[34]:


car_data.head()


# In[35]:


# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'Brand' column to convert it to numeric
car_data['brand'] = label_encoder.fit_transform(car_data['brand'])


# In[36]:


car_data.head()


# In[37]:


#car_data['model_year'] = label_encoder.fit_transform(car_data['model_year'])
car_data['fuel_type'] = label_encoder.fit_transform(car_data['fuel_type'])
#car_data['transmission'] = label_encoder.fit_transform(car_data['transmission'])
#car_data['ext_col'] = label_encoder.fit_transform(car_data['ext_col'])
#car_data['int_col'] = label_encoder.fit_transform(car_data['int_col'])
car_data['accident'] = label_encoder.fit_transform(car_data['accident'])
#car_data['model'] = label_encoder.fit_transform(car_data['model'])




# In[38]:


# Apply One-Hot Encoding for the categorical columns
#car_data_encoded = pd.get_dummies(car_data, columns=['brand','fuel_type', 'transmission', 'ext_col', 'int_col', 'accident'], drop_first=True)

# Optional: If you want to replace the original car_data with the encoded version:
#car_data = car_data_encoded


# In[39]:


car_data.head()


# In[40]:


car_data.info()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
# Initialize the MinMaxScaler
#scaler = MinMaxScaler()

# Normalize all numeric features
#car_data[car_data.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(car_data.select_dtypes(include=['float64', 'int64']))


# In[41]:


input_data=car_data.drop(columns=['price','model_year'])
output_data=car_data["price"]


# SPLITTING THE DATA

# In[42]:


x_train,x_test,y_train,y_test=train_test_split(input_data,output_data,test_size=.2)


# MODEL CREATION-LINEAR REGRESSION
# 

# In[ ]:


model=LinearRegression()


# TRAIN THE MODEL

# In[ ]:


model.fit(x_train,y_train)


# #MODEL EVALUATION

# In[ ]:


# prediction on training data
prediction_on_train=model.predict(x_train)


# In[ ]:


#R-SQUARED ERROR
error_score=metrics.r2_score(y_train,prediction_on_train)
print("R SQUARED ERROR : ",error_score)


# In[ ]:


car_data.columns


# In[ ]:


from sklearn.metrics import mean_squared_error


# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_train,prediction_on_train )

# Calculate RMSE by taking the square root of MSE
rmse = np.sqrt(mse)

print(f"Root Mean Square Error (RMSE): {rmse}")


# MODEL CREATION-RANDOM FOREST

# In[43]:


from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(x_train, y_train)
y_train_pred = rf.predict(x_train)
y_test_pred = rf.predict(x_test)
train_rmse_rf = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse_rf = mean_squared_error(y_test, y_test_pred, squared=False)
print(f"Train RMSE: {train_rmse_rf}")
print(f"Test RMSE: {test_rmse_rf}\n")
print("R squared for train is :",rf.score(x_train,y_train)*100)
print("R squared for test is :", rf.score(x_test,y_test)*100)


# MODEL CREATION-LINEAR REGRESSION

# In[44]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr = LinearRegression()
lr.fit(x_train, y_train)
y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)
train_rmse_lr = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse_lr = mean_squared_error(y_test, y_test_pred, squared=False)
print(f"Train RMSE: {train_rmse_lr}")
print(f"Test RMSE: {test_rmse_lr}\n")
print("R squared for train is :",lr.score(x_train,y_train)*100)
print("R squared for test is :", lr.score(x_test,y_test)*100)


# MODEL CREATION-DECISION TREE

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(x_train, y_train)
y_train_pred = dt.predict(x_train)
y_test_pred = dt.predict(x_test)
train_rmse_dt = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse_dt = mean_squared_error(y_test, y_test_pred, squared=False)
print(f"Train RMSE: {train_rmse_dt}")
print(f"Test RMSE: {test_rmse_dt}\n")
print("R squared for train is :",dt.score(x_train,y_train)*100)
print("R squared for test is :", dt.score(x_test,y_test)*100)


# In[52]:


car_data_test=pd.read_csv(r"C:\Users\sujil\Downloads\Test.csv")


# In[53]:


car_data_test.head()


# In[54]:


car_data_test.set_index("id",inplace=True)


# In[57]:


car_data_test.isnull().sum()


# In[56]:


#car_data_test.dropna(inplace=True)
car_data_test.fillna(car_data.mean(), inplace=True)


# In[58]:


car_data_test.duplicated().sum()


# In[59]:


car_data_test.drop(columns=["engine"],inplace=True)


# In[60]:


#Q1 = car_data_test_1.quantile(0.25)
#Q3 = car_data_test_1.quantile(0.75)

# Calculate the IQR
#IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
#lower_bound = Q1 - 1.5 * IQR
#upper_bound = Q3 + 1.5 * IQR

# Remove outliers from the dataset
#car_data_test = car_data_test_1[~((car_data_test_1 < lower_bound) | (car_data_test_1 > upper_bound)).any(axis=1)]


# In[61]:


car_data_test.head()


# In[62]:


car_data_test.shape


# In[63]:


#car_data_test_1.shape


# In[64]:


for col in car_data_test.columns:
    print("UNIQUE VALUE OF " + col)
    print(car_data_test[col].unique())
    print("##################################")


# In[65]:


car_data_test.drop(columns=["clean_title"],inplace=True)


# In[66]:


current_year = datetime.now().year
car_data_test['car_age'] = current_year - car_data_test['model_year']
car_data_test.head()


# In[67]:


# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'Brand' column to convert it to numeric
car_data_test['brand'] = label_encoder.fit_transform(car_data_test['brand'])
car_data_test['accident'] = label_encoder.fit_transform(car_data_test['accident'])
car_data_test['fuel_type'] = label_encoder.fit_transform(car_data_test['fuel_type'])


# In[68]:


# Calculate the overall average price from the training data
overall_avg_price = car_data_1['price'].mean()

# For 'model' column in car_data_test
model_price_map = car_data_1.groupby('model')['price'].mean()  # Mean price per model
car_data_test['model'] = car_data_test['model'].map(model_price_map).fillna(overall_avg_price)  # Fill missing with overall avg

# For 'transmission' column in car_data_test
transmission_price_map = car_data_1.groupby('transmission')['price'].mean()  # Mean price per transmission
car_data_test['transmission'] = car_data_test['transmission'].map(transmission_price_map).fillna(overall_avg_price)  # Fill missing

# For 'ext_col' (exterior color) column in car_data_test
ext_col_price_map = car_data_1.groupby('ext_col')['price'].mean()  # Mean price per ext_col
car_data_test['ext_col'] = car_data_test['ext_col'].map(ext_col_price_map).fillna(overall_avg_price)  # Fill missing

# For 'int_col' (interior color) column in car_data_test
int_col_price_map = car_data_1.groupby('int_col')['price'].mean()  # Mean price per int_col
car_data_test['int_col'] = car_data_test['int_col'].map(int_col_price_map).fillna(overall_avg_price)  # Fill missing


# In[69]:


car_data_test.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
# Initialize the MinMaxScaler
#scaler = MinMaxScaler()

# Normalize all numeric features
#3car_data_test[car_data_test.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(car_data_test.select_dtypes(include=['float64', 'int64']))


# In[70]:


input_data_test=car_data_test.drop(columns=['model_year'])


# In[71]:


# prediction on test data
prediction_on_test=rf.predict(input_data_test)


# In[72]:


print(prediction_on_test)


# In[73]:


import pandas as pd

# Assuming prediction_on_test is a 1D array or DataFrame with predictions
predicted_price = pd.DataFrame(prediction_on_test)

# Accessing the index as 'id'
test_id = car_data_test.index  # This should be continuous if set up correctly

# Converting the index to a Series and naming it 'id'
test_id = pd.Series(test_id, name='id')

# Concatenating along the columns
submission = pd.concat([test_id, predicted_price], axis=1)

# Naming the columns
submission.columns = ['id', 'predicted_price']

# Displaying info about the submission DataFrame
submission.info()


# In[74]:


submission.to_csv(r"C:\Users\sujil\OneDrive\Documents\submission2.csv" , index = False)


# In[75]:


r=pd.read_csv(r"C:\Users\sujil\OneDrive\Documents\submission2.csv" )


# In[76]:


r.head()


# In[77]:


import os
print(os.getcwd())



# In[ ]:




