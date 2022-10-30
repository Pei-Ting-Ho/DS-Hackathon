#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from surprise import accuracy
from surprise.reader import Reader
from surprise.dataset import Dataset
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms.matrix_factorization import SVD


# In[2]:


def create_mf_model(df, cid_col, iid_col, iid_colname, cid_colname, response_colname, response_map, response_scale, param_grid=None, grid_search=None, model_params=None):
    
    all_cols = cid_col + iid_col
    interaction_df = df[all_cols].set_index(*cid_col)
    
    interaction_df_melted = interaction_df.unstack().reset_index().rename(columns={'level_0': iid_colname, 'level_1': cid_colname, 0: response_colname})
    interaction_df_melted = interaction_df_melted.dropna(axis=0)
    
    le = LabelEncoder()
    interaction_df_melted[iid_colname] = interaction_df_melted[[iid_colname]].apply(le.fit_transform)
    interaction_df_melted[response_colname] = interaction_df_melted[[response_colname]].replace(response_map)
    
    reader = Reader(rating_scale=response_scale)
    surprise = Dataset.load_from_df(interaction_df_melted[[cid_colname, iid_colname, response_colname]], reader)
    surprise_TRAIN, surprise_TEST = train_test_split(surprise, test_size=0.2, random_state=1)
    
    if grid_search:
        if param_grid:
            
            grid_obj = GridSearchCV(SVD, param_grid=param_grid, measures=['rmse'], cv=3, n_jobs=-1)
            grid_obj.fit(surprise)
            
            params = {}
            for param in param_grid:
                params[param] = grid_obj.best_params['rmse'][param]
            print(params)
                
            model = SVD(**params, verbose=False)
    else:
        if model_params:
            model = SVD(**model_params)
        else:
            model = SVD()
        
    model.fit(surprise_TRAIN)
    predictions = model.test(surprise_TEST)
    accuracy.rmse(predictions)
    
    return interaction_df_melted, le, model 


# In[3]:


def make_response_prediction(df, le, model, cid_colname, iid_colname, response_colname):
        
    interaction_matrix = df.pivot(index=cid_colname, columns=iid_colname, values=response_colname)
    
    for customer_id in interaction_matrix.index:
        non_interacted_questions = interaction_matrix.loc[customer_id][interaction_matrix.loc[customer_id].isnull()].index.tolist()
        
        for question_id in non_interacted_questions:
            prediction_id = model.predict(customer_id, question_id).est
            interaction_matrix.loc[customer_id, question_id] = prediction_id
    
    converted_cols = interaction_matrix.columns
    original_cols = le.inverse_transform(interaction_matrix.columns)
    interaction_matrix_fn = interaction_matrix.rename(columns=dict(zip(converted_cols, original_cols))).reset_index(level=cid_colname)
    interaction_matrix_fn.columns.name = None
    
    return interaction_matrix_fn


# In[4]:


def finalize_matrix_completion(df, interaction_df, non_iid_cols):
        
    partition_1 = df[non_iid_cols]
    partition_2 = interaction_df
    df_preprocessed = pd.concat([partition_1, partition_2], axis=1)
    
    return df_preprocessed

