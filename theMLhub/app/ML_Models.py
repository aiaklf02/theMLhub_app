import io
import time
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, classification_report, \
    roc_auc_score, roc_curve, matthews_corrcoef, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO
import base64
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

from sklearn.neural_network import MLPRegressor

from .models import Result
from .visualisation_plots import generate_visualizations, generate_classification_report_plot, \
    generate_confusion_matrix_plot


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,mean_squared_error, mean_absolute_error, r2_score
import time

# remove none values from dict
def remove_none_values(d):
    return {k: v for k, v in d.items() if v is not None}


def encode_categorical_data(data, supervised=None):
    """
    Encodes categorical data based on whether it's supervised or unsupervised.

    Args:
    - data (tuple or pd.DataFrame): If supervised, should be a tuple (X_train, X_test, y_train, y_test).
                                      If unsupervised, should be a DataFrame (df).
    - supervised (bool or None): If True, process data for supervised learning.
                                 If False, process for unsupervised learning.
                                 If None, do nothing.

    Returns:
    - Processed data (X_train, X_test, y_train, y_test or df).
    """
    if supervised is True:
        # Handle supervised case: (X_train, X_test, y_train, y_test)
        X_train, X_test, y_train, y_test = data

        # Encode categorical features in X_train and X_test
        for col in X_train.select_dtypes(include=['object']).columns:
            X_train[col] = X_train[col].astype('category').cat.codes
            X_test[col] = X_test[col].astype('category').cat.codes

        return X_train, X_test, y_train, y_test

    elif supervised is False:
        # Handle unsupervised case: DataFrame (df)
        if isinstance(data, tuple):
            raise ValueError("For unsupervised learning, the input data must be a DataFrame, not a tuple.")

        df = data

        # Encode categorical features in the entire dataframe
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes

        return df

    elif supervised is None:
        # Do nothing if supervised is None
        return data
    else:
        raise ValueError("Invalid value for 'supervised'. It must be True, False, or None.")


def train_linear_regression(preprocesseddata, params, target_column=None):

    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        # Track training time
        start_train_time = time.time()
        #
        print('training started')
        if 'n_jobs' in params.keys():
            n_jobs = int(params['n_jobs'])
        else:
            n_jobs = 3

        model = LinearRegression(n_jobs=n_jobs)
        model.fit(X_train, y_train)
        #
        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        # Track testing (prediction) time
        start_test_time = time.time()
        #
        predictions = model.predict(X_test)
        #
        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        # Generate plots
        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        metric_results = {
            "mse": mean_squared_error(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
            "training time": training_time,
            "testing time": testing_time,
        }


        obj = {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,  # Include the trained model
        }

        return obj

    else:
        raise Exception('linear regression require target Column !')

# lightgbm regression
def train_regression_LightGBM(preprocesseddata, params, target_column=None):
    print(f" i got these params from front end : {params}.\n")

    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        # Track training time
        start_train_time = time.time()
        print('training started')

        # Create the LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        if 'auto' in params.keys() and params['auto'] == "true":
            params = None
            params = {
                'objective': 'regression',  # For regression tasks
                'metric': 'l2',  # Mean squared error
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.01,
                'feature_fraction': 0.9,
            }

        # Train the model
        model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        # Track testing (prediction) time
        start_test_time = time.time()

        # Predict on the test set
        predictions = model.predict(X_test, num_iteration=model.best_iteration)

        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        # Generate plots (optional)
        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        # Calculate regression metrics
        metric_results = {
            "mse": mean_squared_error(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
            "training time": training_time,
            "testing time": testing_time,
        }

        obj = {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,  # Include the trained model
        }

        return obj
    else:
        raise Exception('Target column is required for LightGBM Regression.')




# Decision Tree
def decisionTreeCart(preprocesseddata, params, target_column=None):
    print(f" i got these params from front end : {params}.\n")
    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        start_train_time = time.time()
        print('Decision Tree training started')

        model = DecisionTreeRegressor(
            max_depth=int(params.get("max_depth",10)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1))
        )
        model.fit(X_train, y_train)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        start_test_time = time.time()
        predictions = model.predict(X_test)
        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        metric_results = {
            "mse": mean_squared_error(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
            "training time": training_time,
            "testing time": testing_time,
        }

        return {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,
        }
    else:
        raise Exception('Target column is required for Decision Tree Regression.')


# Random Forest
def train_random_forest(preprocesseddata, params, target_column=None):
    print(f" i got these params from front end : {params}.\n")
    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        start_train_time = time.time()
        print('Random Forest training started')

        model = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth",10)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1))
        )
        model.fit(X_train, y_train)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        start_test_time = time.time()
        predictions = model.predict(X_test)
        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        metric_results = {
            "mse": mean_squared_error(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
            "training time": training_time,
            "testing time": testing_time,
        }

        return {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,
        }
    else:
        raise Exception('Target column is required for Random Forest Regression.')


# KNN
def train_knn(preprocesseddata, params, target_column=None):
    print(f" i got these params from front end : {params}.\n")
    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        start_train_time = time.time()
        print('KNN training started')

        model = KNeighborsRegressor(
            n_neighbors=int(params.get("n_neighbors", 5)),
            weights=params.get("weights", "uniform"),
            metric=params.get("metric", "minkowski")
        )
        model.fit(X_train, y_train)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        start_test_time = time.time()
        predictions = model.predict(X_test)
        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        metric_results = {
            "mse": mean_squared_error(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
            "training time": training_time,
            "testing time": testing_time,
        }

        return {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,
        }
    else:
        raise Exception('Target column is required for KNN Regression.')


# SVR
def train_svr(preprocesseddata, params, target_column=None):
    print(f" i got these params from front end : {params}.\n")
    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        start_train_time = time.time()
        print('SVR training started')

        model = SVR(
            C=int(params.get("C", 1.0)),
            epsilon=float(params.get("epsilon", 0.1)),
            kernel=params.get("kernel", "rbf")
        )
        model.fit(X_train, y_train)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        start_test_time = time.time()
        predictions = model.predict(X_test)
        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        metric_results = {
            "mse": mean_squared_error(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
            "training time": training_time,
            "testing time": testing_time,
        }

        return {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,
        }
    else:
        raise Exception('Target column is required for SVR.')


# XGBoost
def train_xgboost(preprocesseddata, params, target_column=None):
    print(f" i got these params from front end : {params}.\n")

    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        start_train_time = time.time()
        print('XGBoost training started')

        model = xgb.XGBRegressor(
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 6)),
            n_estimators=int(params.get("n_estimators", 100))
        )
        model.fit(X_train, y_train)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        start_test_time = time.time()
        predictions = model.predict(X_test)
        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        metric_results = {
            "mse": mean_squared_error(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
            "training time": training_time,
            "testing time": testing_time,
        }

        return {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,
        }
    else:
        raise Exception('Target column is required for XGBoost Regression.')


# Neural Network (MLP)

def train_reseau_neuron(preprocesseddata, params, target_column=None):
    print(f" i got these params from front end : {params}.\n")

    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        start_train_time = time.time()
        print('Neural Network training started')

        hidden_layer_sizes = tuple(map(int, params.get("hidden_layer_sizes", "100").split(',')))
        activation = params.get("activation", "relu")
        solver = params.get("solver", "adam")

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=int(params.get("max_iter", 200))
        )
        model.fit(X_train, y_train)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        start_test_time = time.time()
        predictions = model.predict(X_test)
        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        metric_results = {
            "mse": mean_squared_error(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
            "training time": training_time,
            "testing time": testing_time,
        }

        return {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,
        }
    else:
        raise Exception('Target column is required for Neural Network Regression.')



def KMeansClustering(preprocesseddata, params, target_column=None):
    print(f" i got these params from front end : {params}.\n")

    try:
        # Encode the categorical data (if needed) for unsupervised learning
        df = encode_categorical_data(preprocesseddata, supervised=False)

        # Track training time
        start_train_time = time.time()
        print('training started kmeans')

        if 'auto' in params.keys() and params['auto'] == 'true':
            n_clusters = params.get('n_clusters', 3)
            max_iter = params.get('max_iter', 1000)
        else:
            n_clusters = 3
            max_iter = 1000

        print(f'running kmeans with nclusters = {n_clusters}\n max iter {max_iter}')
        # Train the KMeans model
        model = KMeans(n_clusters=n_clusters, max_iter=max_iter,random_state=42)
        model.fit(df)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        # Track testing (prediction) time
        start_test_time = time.time()

        # Predict on the test set (assign cluster labels to the data)
        cluster_labels = model.predict(df)

        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        # Generate plots (optional)
        plots = generate_visualizations(df, df, None, None, model)

        # Calculate clustering metrics
        silhouette = silhouette_score(df, cluster_labels)

        metric_results = {
            "silhouette_score": silhouette,
            "training time": training_time,
            "testing time": testing_time,
        }

        obj = {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,  # Include the trained model
        }

        return obj
    except Exception as e:
        raise e


def train_logistic_regression(preprocesseddata, params, target_column=None):

    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        # Track training time
        start_train_time = time.time()
        print('training started')
        #
        if 'max_iter' in params.keys():
            max_iter = int(params['max_iter'])
        else:
            max_iter = 1000
        # Train model
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)
        #
        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        # Track testing (prediction) time
        start_test_time = time.time()
        #
        # Evaluate model
        predictions = model.predict(X_test)
        #
        end_test_time = time.time()
        testing_time = end_test_time - start_test_time


        # Calculate classification metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
        f1 = f1_score(y_test, predictions, average='weighted')

        try:
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # Only for binary classification
        except ValueError:
            auc = None

        # Generate plots
        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        plots['confusion_matrix'] = generate_confusion_matrix_plot(y_test, predictions, model)

        metric_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "training time": training_time,
            "testing time": testing_time,
        }

        obj = {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,  # Include the trained model

        }

        return obj
    else:
        raise Exception('Target column is required for Logistic Regression.')

# lightgbm classification
def train_classification_LightGBM(preprocesseddata, params, target_column=None):
    print(f" i got these params from front end : {params}.\n")

    # Determine objective: multiclass or binary
    if params.get('auto', False):  # Safely check for 'auto' key
        objective = 'multiclass'
        hyperparameters = {}  # Default to empty dict when auto is True
    else:
        hyperparameters = params.get('hyperparameters', {})  # Handle missing hyperparameters
        objective = 'binary' if hyperparameters.get('num_class', 2) == 2 else 'multiclass'
        print(objective)

    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata,supervised=True)

        # Track training time
        start_train_time = time.time()
        print('training started')

        # Create the LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            'objective': 'binary' if objective == 'binary' else 'multiclass',
            'metric': 'binary_error' if objective == 'binary' else 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': hyperparameters.get('nLeaves', 31),
            'learning_rate': hyperparameters.get('learningRate', 0.05),
            'feature_fraction': 0.9,
        }

        if objective == 'multiclass':
            params['num_class'] = hyperparameters.get('num_class', 3)

        try:
            # Train the model
            model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

            end_train_time = time.time()
            training_time = end_train_time - start_train_time

            # Track testing (prediction) time
            start_test_time = time.time()

            # Predict on the test set
            predictions = model.predict(X_test, num_iteration=model.best_iteration)

            # Convert probabilities to binary labels for classification
            predictions_binary = (predictions >= 0.5).astype(int) if objective == 'binary' else predictions.argmax(axis=1)

            end_test_time = time.time()
            testing_time = end_test_time - start_test_time

            # Generate plots (optional)
            plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

            # Calculate classification metrics
            accuracy = accuracy_score(y_test, predictions_binary)
            precision = precision_score(y_test, predictions_binary, average='binary' if objective == 'binary' else 'macro')
            recall = recall_score(y_test, predictions_binary, average='binary' if objective == 'binary' else 'macro')
            f1 = f1_score(y_test, predictions_binary, average='binary' if objective == 'binary' else 'macro')

            auc = None
            if objective == 'binary':
                try:
                    auc = roc_auc_score(y_test, predictions)
                except ValueError:
                    auc = None

            metric_results = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1 score": f1,
                "auc": auc,
                "training time": training_time,
                "testing time": testing_time,
            }

            obj = {
                "metric_results": remove_none_values(metric_results),
                "plots": remove_none_values(plots),
                "model": model,  # Include the trained model
            }

            return obj
        except Exception as e:
            raise Exception(f'{e}')
    else:
        raise Exception('Target column is required for LightGBM Classification.')

