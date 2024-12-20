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
from .models import Result
from .visualisation_plots import generate_visualizations, generate_classification_report_plot, \
    generate_confusion_matrix_plot

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
        model = LinearRegression()
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

def train_logistic_regression(preprocesseddata, params, target_column=None):

    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        # Track training time
        start_train_time = time.time()
        print('training started')
        #
        # Train model
        model = LogisticRegression(max_iter=2000)
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

        # Specify the parameters for regression
        # params = {
        #     'objective': 'regression',  # For regression tasks
        #     'metric': 'l2',  # Mean squared error
        #     'boosting_type': 'gbdt',
        #     'num_leaves': 31,
        #     'learning_rate': 0.01,
        #     'feature_fraction': 0.9,
        # }

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

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def KMeansClustering(preprocesseddata, params, target_column=None):
    try:
        # Encode the categorical data (if needed) for unsupervised learning
        df = encode_categorical_data(preprocesseddata, supervised=False)

        # Track training time
        start_train_time = time.time()
        print('training started kmeans')

        if params and "n_clusters" in params:
            n_clusters = params.get('n_clusters', 3)
        else:
            n_clusters = 3
        # Train the KMeans model
        model = KMeans(n_clusters=n_clusters, random_state=42)
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


# lightgbm classification
def train_classification_LightGBM(preprocesseddata, params,target_column=None):
    objective = params['objective']
    # objective : multiclass or binary

    if target_column:
        X_train, X_test, y_train, y_test = preprocesseddata

        # Track training time
        start_train_time = time.time()
        print('training started')

        # Create the LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        if objective == 'binary':
            # For binary classification
            params = {
                'objective': 'binary',
                'metric': 'binary_error',  # Binary classification
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
            }
        else:
            # For multi-class classification
            params = {
                'objective': 'multiclass',
                'metric': 'multi_logloss',  # Multi-class classification
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'num_class': 3,  # Specify the number of classes for multi-class classification
            }


        # Train the model
        model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100, early_stopping_rounds=10)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        # Track testing (prediction) time
        start_test_time = time.time()

        # Predict on the test set
        predictions = model.predict(X_test, num_iteration=model.best_iteration)

        # Convert probabilities to binary labels for classification
        predictions_binary = (predictions >= 0.5).astype(int)

        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        # Generate plots (optional)
        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, predictions_binary)
        precision = precision_score(y_test, predictions_binary)
        recall = recall_score(y_test, predictions_binary)
        f1 = f1_score(y_test, predictions_binary)

        try:
            auc = roc_auc_score(y_test, predictions)  # Only for binary classification
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
    else:
        raise Exception('Target column is required for LightGBM Classification.')
