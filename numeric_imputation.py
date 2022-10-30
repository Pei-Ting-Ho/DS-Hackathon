#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


# In[2]:


def replace_with_median(df, existed_variables):
    
    df_imputed = df.copy()
    vars_median = dict(df_imputed[existed_variables].median())
    df_imputed[existed_variables] = df_imputed[existed_variables].fillna(vars_median)
    
    return df_imputed


# In[3]:


def replace_with_regression(df, existed_variables):
    
    df_imputed = df.copy()
    X, y = existed_variables
    
    mask_notnull = ((~df_imputed[X].isnull()) & (~df_imputed[y].isnull()))
    X_TRAIN = df_imputed[mask_notnull][[X]]
    y_TRAIN = df_imputed[mask_notnull][[y]]

    lr = LinearRegression()
    lr.fit(X_TRAIN, y_TRAIN)
    print(f'score: {lr.score(X_TRAIN, y_TRAIN)}')
    print(f'coef: {lr.coef_}')
    
    mask_null = ((~df_imputed[X].isnull()) & (df_imputed[y].isnull()))
    X_TRAIN_pred = df_imputed[mask_null][[X]]
    y_TRAIN_pred = lr.predict(X_TRAIN_pred).flatten()
     
    df_imputed.loc[mask_null, y] = y_TRAIN_pred
    
    return df_imputed


# In[4]:


def replace_with_regression_and_median(df, existed_variables):
        
    df_imputed = df.copy()
    X, y = existed_variables
    
    mask_notnull = ((~df_imputed[X].isnull()) & (~df_imputed[y].isnull()))
    X_TRAIN = df_imputed[mask_notnull][[X]]
    y_TRAIN = df_imputed[mask_notnull][[y]]

    lr = LinearRegression()
    lr.fit(X_TRAIN, y_TRAIN)
    print(f'score: {lr.score(X_TRAIN, y_TRAIN)}')
    print(f'coef: {lr.coef_}')
    
    mask_null = ((~df_imputed[X].isnull()) & (df_imputed[y].isnull()))
    X_TRAIN_pred = df_imputed[mask_null][[X]]
    y_TRAIN_pred = lr.predict(X_TRAIN_pred).flatten()
     
    df_imputed.loc[mask_null, y] = y_TRAIN_pred
    
    df_imputed = replace_with_median(df=df_imputed, existed_variables=existed_variables)
    
    return df_imputed


# In[5]:


def create_feature(df, replace_func, existed_variables, mutated_variable):
    
    df_imputed = replace_func(df, existed_variables)
    
    if len(existed_variables) == 2:
        df_imputed[mutated_variable] = df_imputed[existed_variables[0]] + df_imputed[existed_variables[1]]
        df_preprocessed = df_imputed.drop(existed_variables, axis=1)
        
    return df_preprocessed

