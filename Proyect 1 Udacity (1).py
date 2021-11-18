#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer

#Gather
    df = pd.read_excel("C:/Users/Teletrabajo/Documents/DataScience/Proyecto 1/Data_Extract_From_Gender_Statistics (1).xlsx",  sheet_name="Data")

    df2 = df.transpose()
    df2.head()



    


# In[46]:


#1)  Look at the data Assess
    no_nulls = set(df2.columns[df2.isnull().mean()==0])
    print(no_nulls)


#Provide the number of rows in the dataset

    num_rows = df2.shape[0] 
    print(str("Numbers of rows are ") + str(num_rows))

#Provide the number of columns in the dataset

    num_cols = df2.shape[1]
    print(str("Numbers of columns are ") + str(num_cols))

#Types 
     
    print(df2.dtypes)

#Basic Stadistics

    stadistics = df2.describe()
    print(stadistics)


# In[47]:


#2) Clean Dataset

#a) Drop 2, 3, and 4 rows

    df2.drop(["Series Code","Country Name","Country Code"], axis=0)

#Rename Columns

#Create a list with the columns names

    Columns = list(df2.iloc[0])
    print(Columns)

#Assing the column names directly

    df2.columns = Columns

    df2.head()

#Drop all rows without use

    df2 = df2.drop(["Series Name","Series Code","Country Name","Country Code" ], axis=0)


# In[48]:


#Drop columns with >75% NaN's values

    print(df2.isnull().mean() >= 0.50)

#Drop Columns

    df3 = df2.drop(['Location of cooking: inside the house (% of households)',
'Location of cooking: outdoors (% of households)',
'Location of cooking: other places (% of households)',
'Main cooking fuel: dung (% of households)',
'Location of cooking: separate building (% of households)',
'Main cooking fuel: agricultural crop (% of households)',
'Main cooking fuel: straw/shrubs/grass (% of households)',
'Poverty headcount ratio at national poverty lines (% of population)', 
'Households with water 30 minutes or longer away round trip (%)',
'Households with water less than 30 minutes away round trip (%)',
'Households with water on the premises (%)',
'Children in employment, female (% of female children ages 7-14)',                       
'Children in employment, male (% of male children ages 7-14)',                           
'Children in employment, total (% of children ages 7-14)',
'Main cooking fuel: agricultural crop (% of households)',
'Main cooking fuel: straw/shrubs/grass (% of households)',
'Main cooking fuel: LPG/natural gas/biogas (% of households)',
'Dismissal of pregnant workers is prohibited (1=yes; 0=no)',
'A woman can get a job in the same way as a man (1=yes; 0=no)',
'A woman can work at night in the same way as a man (1=yes; 0=no)',
'A woman can work in an industrial job in the same way as a man (1=yes; 0=no)',
'Main cooking fuel: wood (% of households)',                                            
'Main cooking fuel: electricity  (% of households)',                                     
'Main cooking fuel: dung (% of households)',                                             
'Main cooking fuel: charcoal (% of households)',                                         
'Location of cooking: separate building (% of households)',                              
'Main cooking fuel: agricultural crop (% of households)',
'Main cooking fuel: straw/shrubs/grass (% of households)',                               
'Main cooking fuel: LPG/natural gas/biogas (% of households)',                           
'Poverty headcount ratio at national poverty lines (% of population)'], axis = 1)


# In[54]:


#Working with Categorical Variables

    df3 = pd.get_dummies(df3, columns = ["A woman can work in a job deemed dangerous in the same way as a man (1=yes; 0=no)"],prefix = "Column")

    df3.head(n=20)


# In[50]:


# Column with NaN Values

    nulls = set(df3.columns[df3.isnull().mean()!=0])
    print(nulls)


# In[51]:


#Cambiar el formato de los números (todo está como string)

    df3 = df3.astype({'GDP (current US$)':'float', 'GDP growth (annual %)': 'float', 
                  'GDP per capita (constant 2010 US$)':'float',
                  'GDP per capita (Current US$)' : 'float',
                  'GNI per capita, Atlas method (current US$)' : 'float',
                  'Gini index (World Bank estimate)' : 'float',
                  'GNI per capita, PPP (current international $)' : 'float',
                  'GNI, Atlas method (current US$)' : 'float',
                  'Immunization, DPT (% of children ages 12-23 months)' : 'float',
                  'Inflation, consumer prices (annual %)' : 'float',
                  'Immunization, measles (% of children ages 12-23 months)' : 'float',
                  'Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population)' : 'float',
                  'People practicing open defecation, urban (% of urban population)' : 'float',
                  'People practicing open defecation, rural (% of rural population)' : 'float',
                  'People practicing open defecation (% of population)' : 'float',
                  'Employers, female (% of female employment) (modeled ILO estimate)' : 'float',
                  'Employers, male (% of male employment) (modeled ILO estimate)' : 'float',
                  'Employers, total (% of total employment) (modeled ILO estimate)' : 'float',
    })


# In[52]:





# In[53]:


print(df3.dtypes)


# In[60]:


df3.head(n=20)


# In[89]:


#Llenar los Nan con las medias de cada uno de los valores

    fill_mean = lambda col: col.fillna(col.mean()) # Mean function

    fill_df3 = df3.apply(fill_mean, axis=0) #Fill all missing values with the mean of the column.

    fill_df3.head(n=20)


# In[104]:


#Linear Regression 

#a) Split into Explanatory and response variables

    X = fill_df3.drop(['GDP (current US$)'], axis= 1)
    y = fill_df3['GDP (current US$)']

    print(X)
    print(y)
                      
#Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Linear Regression Model

    lm_model = LinearRegression(normalize=True) # Instantiate
    lm_model.fit(X_train, y_train) #Fit

    #Predict using your model
    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)

    #Score using your model
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)

    print(test_score)
    print(train_score)


# In[107]:





# In[ ]:




