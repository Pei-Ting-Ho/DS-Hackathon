# DS-Hackathon

## Problem Statement
This machine learning exercise aims to:
- Predict whether a passenger was satisfied or not considering the overall experience of traveling on the Shinkansen Bullet Train
- Determine the relative importance of parameters with regards to their contribution to the passengers' overall travel experience

The problem consists of separate datasets: 
- Travel data, includes the passenger information and the train attributes. 
- Survey data, includes the survey responses of passengers related to the post-service experience.

## Model Files
- Step 1: **Preliminary_Analysis**, contains the data inspection, data preprocessing, and data exploration.
- Step 2: **Preprocessing_Setups_Illustration**, experiments with different imputation setups, both numeric and categorical.
  - categorical_imputation.py
  - numeric_imputation.py
- Step 3.1: **Model_Experiments**, leverages the AutoML package ([PyCaret](https://pycaret.gitbook.io/docs/)) to perform the modelling task.
- Step 3.2: **Model_Fn**, finalizes the model predictions, based on the selected imputation method and the stacking ensemble classifiers.

## Leaderboard Performance
- Highest Accuracy: 0.9602
- Final Model: Stacker (CatBoost Classifier	+ Extreme Gradient Boosting	+ Random Forest Classifier, with each model tuned with Bayesian Optimization)
- Rank: 2 / 60+

