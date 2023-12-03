"""

Rohan Narasayya
CSE 163 Aj
This program uses the Bikeshare data about the Capital Bikeshare
system in Washington D.C over a two-year period from 2011 to 2012
to predict total, casual, and registered ridership.
"""

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt


def predict_total_ridership(df):
    """
    This function takes a dataframe and predicts the total
    ridership.
    """
    df = df.dropna()
    df_cnt = df.drop(['casual', 'registered'], axis=1)
    features = df_cnt.loc[:, df_cnt.columns != 'cnt']
    labels = df_cnt['cnt']
    features = pd.get_dummies(features)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    parameters = { "n_neighbors": range(1, 10),
                   "weights": ["uniform", "distance"], }
    gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
    gridsearch.fit(features_train, labels_train)
    GridSearchCV(estimator=KNeighborsRegressor(),
                param_grid={'n_neighbors': range(1, 10),
                    'weights': ['uniform', 'distance']})
    gridsearch.best_params_
    test_preds_grid = gridsearch.predict(features_test)
    test_mse = mean_squared_error(labels_test, test_preds_grid)
    test_rmse = sqrt(test_mse)
    print('The root mean square error of predicting total'
          ' ridership is ' + str(test_rmse) + '.')
    print('For reference, the mean and standard deviation of'
          ' cnt are ' + str(df['cnt'].mean()) +
          ' and ' + str(df['cnt'].std()))


def predict_casual_ridership(df):
    """
    This function takes a dataframe and predicts the casual
    ridership.
    """
    df = df.dropna()
    df_cas = df.drop(['registered', 'cnt'], axis=1)
    features = df_cas.loc[:, df_cas.columns != 'casual']
    labels = df_cas['casual']
    features = pd.get_dummies(features)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    parameters = { "n_neighbors": range(1, 10),
                   "weights": ["uniform", "distance"], }
    gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
    gridsearch.fit(features_train, labels_train)
    GridSearchCV(estimator=KNeighborsRegressor(),
                param_grid={'n_neighbors': range(1, 10),
                    'weights': ['uniform', 'distance']})
    gridsearch.best_params_
    test_preds_grid = gridsearch.predict(features_test)
    test_mse = mean_squared_error(labels_test, test_preds_grid)
    test_rmse = sqrt(test_mse)
    print('The root mean squared error of predicting casual'
          ' ridership is ' + str(test_rmse) + '.')
    print('For reference, the mean and standard deviation'
          ' of casual are ' + str(df['casual'].mean()) 
          + ' and ' + str(df['casual'].std()))


def predict_registered_ridership(df):
    """
    This function takes a dataframe and predicts the registered
    ridership.
    """
    df = df.dropna()
    df_registered = df.drop(['casual', 'cnt'], axis=1)
    not_registered = df_registered.columns != 'registered'
    features = df_registered.loc[:, not_registered]
    labels = df_registered['registered']
    features = pd.get_dummies(features)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    parameters = { "n_neighbors": range(1, 10),
                   "weights": ["uniform", "distance"], }
    gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
    gridsearch.fit(features_train, labels_train)
    GridSearchCV(estimator=KNeighborsRegressor(),
                param_grid={'n_neighbors': range(1, 10),
                    'weights': ['uniform', 'distance']})
    gridsearch.best_params_
    test_preds_grid = gridsearch.predict(features_test)
    test_mse = mean_squared_error(labels_test, test_preds_grid)
    test_rmse = sqrt(test_mse)
    print('The root mean squared error of predicting registered'
          ' ridership is ' + str(test_rmse) + '.')
    print('For reference, the mean and standard deviation of'
          ' registered are ' + str(df['registered'].mean())
          + ' and ' + str(df['registered'].std()))