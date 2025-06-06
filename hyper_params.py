import proj_setup
import pandas as pd
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, callback
import matplotlib.pyplot as plt
import numpy as np

class CustomSlidingWindow:
    def split(X: pd.DataFrame, y = None, num_windows: int = 3):

        years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]

        for i in range(num_windows):

            training_years = years[i : i + 3]
            validation_year = years[i + 3]

            # bool array telling us which data in X have the years we want
            train_m = X['YEAR'].isin(training_years)
            validation_m = X['YEAR'] == validation_year

            train_indices = np.where(train_m)[0]
            valid_indices = np.where(validation_m)[0]

            yield train_indices, valid_indices

# initializing ...
cbb_df = proj_setup.read('./data/cbb.csv', './data/cbb25.csv')

cbb_df = proj_setup.encode_postseason(cbb_df)

cbb_df = proj_setup.encode_conference(cbb_df)

initial_train, initial_test = proj_setup.split_data(cbb_df)

accuracy = []

clf = XGBClassifier(objective='binary:logistic',
                    base_score=0.192,
                    eval_metric='aucpr',
                    subsample=0.8,
                    colsample_bytree=0.8)

pipeline = Pipeline([
    ('classifier', clf)
])

params = {
    'classifier__gamma': [0.2, 0.4, 0.6, 0.8],
    'classifier__max_depth': [2, 3, 4, 5],
    'classifier__min_child_weight': [1, 5, 10, 20, 50],
    'classifier__max_delta_step': [0, 3, 6, 9],
    'classifier__eta': [0.3, 0.4, 0.6, 0.8],
    'classifier__num_parallel_tree': [1, 5, 10, 15]
}

# need to drop postseason
y = initial_train['POSTSEASON']
X = initial_train.drop('POSTSEASON', axis = 1)

# cv: needs an iterable of the splits as arrays of indices
my_cv = CustomSlidingWindow.split(X, y, 7)

#for index, (x,y) in enumerate(my_cv):
#  print(f"Split number: {index}")
#   print(f"Training indices: {x}")
#   print(f"Test indices: {y}\n")

gridsearch = GridSearchCV(estimator = pipeline, param_grid = params, cv = my_cv, n_jobs = -1)

# instead of using multiple for loops to change hyper parameters, we can test them using grid search
# --- START OF MODEL PROGRAM ---

gridsearch.fit(X, y)

print(f'Best Parameters: {gridsearch.best_params_}')
print(f'Best Estimator: {gridsearch.best_estimator_}')
print(f'Best Training Score: {gridsearch.best_score_}')
best_model = gridsearch.best_estimator_

y_initial_test = initial_test['POSTSEASON']
X_initial_test = initial_test.drop('POSTSEASON', axis = 1)

ypred = best_model.predict(X_initial_test)

accuracy.append(accuracy_score(y_initial_test, ypred))

#only use first 3 windows to reduce run time

print(accuracy)
print(np.mean(accuracy))