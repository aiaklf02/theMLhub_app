import h2o
from h2o.estimators import H2OKMeansEstimator
from h2o.automl import H2OAutoML
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,mean_squared_error, \
    mean_absolute_error, r2_score
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


from sklearn.naive_bayes import GaussianNB

def train_classification_naiveBayes(preprocesseddata, params, target_column=None):
    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        # Track training time
        start_train_time = time.time()
        print('training started Naive Bayes')

        # Set smoothing parameter (default is 1.0)
        smoothing = float(params.get('smoothing', 1.0))

        # Train model
        model = GaussianNB(var_smoothing=smoothing)
        model.fit(X_train, y_train)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        # Track testing (prediction) time
        start_test_time = time.time()

        # Evaluate model
        predictions = model.predict(X_test)

        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        # Generate plots (optional)
        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
        f1 = f1_score(y_test, predictions, average='weighted')

        metric_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training time": training_time,
            "testing time": testing_time,
        }

        return {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,  # Include the trained model
        }
    else:
        raise Exception('Target column is required for Naive Bayes classification.')

from sklearn.tree import DecisionTreeClassifier

def train_classification_cart_decision_tree(preprocesseddata, params, target_column=None):
    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        # Track training time
        start_train_time = time.time()
        print('training started Decision Tree')

        # Set hyperparameters
        max_depth = int(params.get('maxDepth', 5))
        min_samples_split = int(params.get('minSamplesSplit', 2))

        # Train model
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        model.fit(X_train, y_train)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        # Track testing (prediction) time
        start_test_time = time.time()

        # Evaluate model
        predictions = model.predict(X_test)

        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        # Generate plots (optional)
        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
        f1 = f1_score(y_test, predictions, average='weighted')

        metric_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training time": training_time,
            "testing time": testing_time,
        }

        return {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,  # Include the trained model
        }
    else:
        raise Exception('Target column is required for Decision Tree classification.')


from sklearn.ensemble import RandomForestClassifier

def train_classification_random_forest(preprocesseddata, params, target_column=None):
    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        # Track training time
        start_train_time = time.time()
        print('training started Random Forest')

        # Set hyperparameters
        n_estimators = int(params.get('nEstimators', 100))
        max_depth = int(params.get('maxDepth', 5))

        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        # Track testing (prediction) time
        start_test_time = time.time()

        # Evaluate model
        predictions = model.predict(X_test)

        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        # Generate plots (optional)
        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
        f1 = f1_score(y_test, predictions, average='weighted')

        metric_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training time": training_time,
            "testing time": testing_time,
        }

        return {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,  # Include the trained model
        }
    else:
        raise Exception('Target column is required for Random Forest classification.')


from sklearn.neighbors import KNeighborsClassifier

def train_classification_knn(preprocesseddata, params, target_column=None):
    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        # Track training time
        start_train_time = time.time()
        print('training started KNN')

        # Set hyperparameters
        n_neighbors = int(params.get('nNeighbors', 5))
        weights = params.get('weights', 'uniform')

        # Train model
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        model.fit(X_train, y_train)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        # Track testing (prediction) time
        start_test_time = time.time()

        # Evaluate model
        predictions = model.predict(X_test)

        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        # Generate plots (optional)
        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
        f1 = f1_score(y_test, predictions, average='weighted')

        metric_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training time": training_time,
            "testing time": testing_time,
        }

        return {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,  # Include the trained model
        }
    else:
        raise Exception('Target column is required for KNN classification.')


from sklearn.svm import SVC

def train_classification_svc(preprocesseddata, params, target_column=None):
    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        # Track training time
        start_train_time = time.time()
        print('training started SVM')

        # Set hyperparameters
        kernel = params.get('kernel', 'rbf')
        C = float(params.get('C', 1.0))

        # Train model
        model = SVC(kernel=kernel, C=C)
        model.fit(X_train, y_train)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        # Track testing (prediction) time
        start_test_time = time.time()

        # Evaluate model
        predictions = model.predict(X_test)

        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        # Generate plots (optional)
        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
        f1 = f1_score(y_test, predictions, average='weighted')

        metric_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training time": training_time,
            "testing time": testing_time,
        }

        return {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,  # Include the trained model
        }
    else:
        raise Exception('Target column is required for SVM classification.')


import xgboost as xgb

def train_classification_xgboost(preprocesseddata, params, target_column=None):
    if target_column:
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        # Track training time
        start_train_time = time.time()
        print('training started XGBoost')

        # Set hyperparameters
        n_estimators = int(params.get('nEstimators', 100))
        max_depth = int(params.get('maxDepth', 5))

        # Train model
        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        # Track testing (prediction) time
        start_test_time = time.time()

        # Evaluate model
        predictions = model.predict(X_test)

        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        # Generate plots (optional)
        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
        f1 = f1_score(y_test, predictions, average='weighted')

        metric_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training time": training_time,
            "testing time": testing_time,
        }

        return {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,  # Include the trained model
        }
    else:
        raise Exception('Target column is required for XGBoost classification.')


def train_classification_reseau_neuron(preprocesseddata, params, target_column=None):
    if target_column:
        # Encode categorical data
        X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

        # Track training time
        start_train_time = time.time()
        print('Training started for Neural Network')

        # Get hyperparameters
        hidden_layer_sizes = tuple(map(int, params.get("hidden_layer_sizes", "100").split(',')))
        activation = params.get("activation", "relu")
        solver = params.get("solver", "adam")
        max_iter = int(params.get("max_iter", 200))

        # Train model
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                              solver=solver, max_iter=max_iter)
        model.fit(X_train, y_train)

        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        # Track testing (prediction) time
        start_test_time = time.time()

        # Evaluate model
        predictions = model.predict(X_test)

        end_test_time = time.time()
        testing_time = end_test_time - start_test_time

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
        f1 = f1_score(y_test, predictions, average='weighted')

        # Generate plots (optional)
        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)

        # Prepare the results
        metric_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training time": training_time,
            "testing time": testing_time,
        }

        return {
            "metric_results": remove_none_values(metric_results),
            "plots": remove_none_values(plots),
            "model": model,  # Include the trained model
        }
    else:
        raise Exception('Target column is required for Neural Network Classification.')



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd

import io
import base64
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import h2o
from h2o.estimators import H2OKMeansEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import io
import base64
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import h2o
from h2o.estimators import H2OKMeansEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import io
import base64
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import h2o
from h2o.estimators import H2OKMeansEstimator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, mean_squared_error, r2_score
from sklearn.metrics import precision_recall_curve


def save_plot_to_base64():
    """Save the current matplotlib plot to a base64-encoded string."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()
    return image_base64

import pandas as pd

def check_task_type(data, target_column):
    """
    Determines if the task is classification or regression based on the target column.

    Args:
    - data (pd.DataFrame or other): The input data containing the features and target column.
    - target_column (str): The name of the target column.

    Returns:
    - str: 'classification' or 'regression' based on the target column type.
    """
    # Attempt to convert the input data to a DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            return {'error': 'Input data could not be converted to a pandas DataFrame.', 'details': str(e)}

    # Now that data is guaranteed to be a DataFrame, proceed with the logic
    try:
        # Get the target column from the dataframe
        target = data[target_column]

        # Check if target column is numeric (for regression)
        if pd.api.types.is_numeric_dtype(target):
            # If numeric, check if it has enough unique values to consider regression
            unique_values = target.nunique()

            # If there are few unique values (like 2 or 3), it's likely classification (binary or multiclass)
            if unique_values <= 10:  # This is an arbitrary threshold, you can adjust it
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'
    except KeyError:
        return {'error': f"Target column '{target_column}' not found in the data."}


def clean_leaderboard(leaderboard):
    cleaned_leaderboard = []
    for entry in leaderboard:
        cleaned_entry = {}
        for key, value in entry.items():
            # Remove underscores from the key
            cleaned_key = key.replace('_', ' ')
            # Remove underscores from the value if it's a string
            if isinstance(value, str):
                cleaned_value = value.replace('_', ' ')
            else:
                cleaned_value = value
            cleaned_entry[cleaned_key] = cleaned_value
        cleaned_leaderboard.append(cleaned_entry)
    return cleaned_leaderboard

def train_h2o_supervised(preprocesseddata, params, target_column):
    # Preprocess the data
    X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)

    # Determine task type (classification or regression)
    task_type = check_task_type(preprocesseddata, target_column)

    # Convert to H2O frame
    train_data = h2o.H2OFrame(X_train)
    train_data[target_column] = h2o.H2OFrame(pd.DataFrame(y_train))  # Convert y_train to DataFrame first
    test_data = h2o.H2OFrame(X_test)
    test_data[target_column] = h2o.H2OFrame(pd.DataFrame(y_test))  # Convert y_test to DataFrame first

    X = [col for col in X_train.columns if col != target_column]
    y = target_column

    # Set AutoML hyperparameters (using values from params if available)
    max_runtime_secs = int(params.get('max_runtime_secs', 60))  # Max time in seconds for training
    max_models = int(params.get('max_models', 3))  # Max models to train
    
    print(f'using *params* max_runtime_secs {max_runtime_secs}\n max_models {max_models}')
    print(f'{params}')
    # Track training time
    start_train_time = time.time()
    print('Training started using H2O AutoML for supervised task')

    # Initialize and train AutoML model
    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        seed=42
    )
    aml.train(x=X, y=y, training_frame=train_data)

    # Get leaderboard and best model
    leaderboard = aml.leaderboard
    print("Leaderboard:\n", leaderboard)
    model = aml.leader

    # Evaluate model performance
    start_test_time = time.time()
    predictions = model.predict(test_data)
    y_pred = predictions.as_data_frame().values.flatten()
    end_test_time = time.time()

    # Handle classification metrics and plots if task is classification
    plots = {}
    if task_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Confusion Matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.levels[1], yticklabels=model.levels[1])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        confusion_matrix_plot_base64 = save_plot_to_base64()
        plots["confusion matrix plot"] = confusion_matrix_plot_base64

        # ROC curve plot for binary classification
        if len(set(y_test)) == 2:  # Only for binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            roc_curve_plot_base64 = save_plot_to_base64()
            plots["roc curve plot"] = roc_curve_plot_base64

        # Precision-Recall curve for imbalanced datasets
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        pr_curve_plot_base64 = save_plot_to_base64()
        plots["precision recall curve plot"] = pr_curve_plot_base64

        mdl = model.model_id.replace('_', ' ')
        metric_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1 score": f1,
            "training time": end_test_time - start_train_time,
            "testing time": end_test_time - start_test_time,
            "model id": mdl,
        }

        leaderboard = leaderboard.as_data_frame().to_dict('records')

        # Handle regression metrics and plots if task is regression
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Residual plot
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        residual_plot_base64 = save_plot_to_base64()
        plots["residual plot"] = residual_plot_base64

        # Prediction vs Actual plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Prediction vs Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        prediction_vs_actual_plot_base64 = save_plot_to_base64()
        plots["prediction vs actual plot"] = prediction_vs_actual_plot_base64

        mdl = model.model_id.replace('_', ' ')

        print(f'plots {plots}')
        metric_results = {
            "mean squared error": mse,
            "r2 score": r2,
            "training time": end_test_time - start_train_time,
            "testing time": end_test_time - start_test_time,
            "model id": mdl,
        }
        leaderboard = leaderboard.as_data_frame().to_dict('records')

    # Replace underscores with spaces in the keys
    metric_results = {k.replace('_', ' '): v for k, v in metric_results.items()}

    # Shut down the H2O cluster
    h2o.shutdown()

    cldrbrd = clean_leaderboard(leaderboard)

    return {
        "metric_results": remove_none_values(metric_results),
        "plots": plots,
        "leaderboard": cldrbrd
    }




def train_h2o_unsupervised(preprocesseddata, params):
    # Preprocess the data (no target column for unsupervised)
    df = encode_categorical_data(preprocesseddata, supervised=False)

    # Convert to H2O frame
    train_data = h2o.H2OFrame(df)

    # Set AutoML hyperparameters (using values from params if available)
    max_runtime_secs = int(params.get('max_runtime_secs', 3600))  # Max time in seconds for training
    n_clusters = int(params.get('nClusters', 3))  # Number of clusters (for KMeans)
    max_iterations = int(params.get('maxIterations', 300))  # Max iterations for KMeans

    # Initialize and train KMeans model for unsupervised learning
    start_train_time = time.time()
    kmeans_model = H2OKMeansEstimator(k=n_clusters, max_iterations=max_iterations, max_runtime_secs=max_runtime_secs)
    kmeans_model.train(training_frame=train_data)

    # Get model details
    model_summary = kmeans_model.summary()

    # Extract relevant metrics from the model and flatten them
    metric_results = {
        "number_of_rows": model_summary['number_of_rows'],
        "number_of_clusters": model_summary['number_of_clusters'],
        "within_cluster_sum_of_squares": model_summary['within_cluster_sum_of_squares'],
        "total_sum_of_squares": model_summary['total_sum_of_squares'],
        "between_cluster_sum_of_squares": model_summary['between_cluster_sum_of_squares'],
    }

    # Extract centroid information and handle both cases
    centroids = kmeans_model.centers()

    # Ensure centroids is a list of H2OFrames or a single H2OFrame
    if isinstance(centroids, list):
        # If centroids is a list, process each centroid as a data frame
        centroids_info = []
        for centroid in centroids:
            if isinstance(centroid, h2o.H2OFrame):
                centroid_info = centroid.as_data_frame().to_dict('records')[0]
                centroids_info.append(centroid_info)
    elif isinstance(centroids, h2o.H2OFrame):
        # If centroids is a single H2OFrame, process it directly
        centroids_info = centroids.as_data_frame().to_dict('records')
    else:
        centroids_info = []

    # Extract scoring history (handle as Pandas DataFrame directly)
    scoring_history = kmeans_model.scoring_history()

    # Convert the Pandas DataFrame to a dictionary for further use
    scoring_history_info = scoring_history.to_dict('records')

    # Flatten the scoring history
    for idx, record in enumerate(scoring_history_info):
        for key, value in record.items():
            metric_results[f"scoring_history_{idx}_{key}"] = value

    # Generate cluster visualization plot
    plt.figure(figsize=(8, 6))
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df)
    for i in range(n_clusters):
        cluster_points = reduced_data[kmeans_model.predict(h2o.H2OFrame(df)).as_data_frame().values.flatten() == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")
    plt.scatter([centroid['x'] for centroid in centroids_info],
                [centroid['y'] for centroid in centroids_info],
                c='red', marker='x', s=100, label='Centroids')
    plt.title('Cluster Visualization (PCA)')
    plt.legend()
    cluster_visualization_plot_base64 = save_plot_to_base64()

    # Track training time
    end_train_time = time.time()
    metric_results["training time"] = end_train_time - start_train_time

    # Replace underscores with spaces in the keys
    metric_results = {k.replace('_', ' '): v for k, v in metric_results.items()}

    # Shut down the H2O cluster
    h2o.shutdown()

    return {
        "metric_results": remove_none_values(metric_results),
        "plots": {
            "cluster_plot": cluster_visualization_plot_base64,
            # Add other plots here as necessary
        }
    }


def train_h2o_automl(preprocesseddata, params, target_column=None):
    paramss = params['hyperparameters']
    h2o.init()

    if target_column:
        # Supervised task (classification or regression)
        result = train_h2o_supervised(preprocesseddata, paramss, target_column)
    else:
        # Unsupervised task (e.g., clustering)
        result = train_h2o_unsupervised(preprocesseddata, paramss)

    return result

# Function to plot the clusters and centroids
def plot_clusters_with_centroids(df, centroids_info, n_clusters, kmeans_model):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df)
    for i in range(n_clusters):
        cluster_points = reduced_data[kmeans_model.predict(h2o.H2OFrame(df)).as_data_frame().values.flatten() == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")
    plt.scatter([centroid['x'] for centroid in centroids_info],
                [centroid['y'] for centroid in centroids_info],
                c='red', marker='x', s=100, label='Centroids')
    plt.title('Cluster Visualization (PCA)')
    plt.legend()

# def train_h2o_automl(preprocesseddata, params, target_column=None):
#     if target_column:
#         # Preprocess the data (ensure correct format for H2O)
#         X_train, X_test, y_train, y_test = encode_categorical_data(preprocesseddata, supervised=True)
#
#         # Start the H2O cluster (only needs to be done once in the program)
#         h2o.init()
#
#         # Convert the Pandas DataFrame to H2O Frame
#         train_data = h2o.H2OFrame(X_train)
#         train_data[target_column] = h2o.H2OFrame(y_train)  # Add target column to the training frame
#         test_data = h2o.H2OFrame(X_test)
#         test_data[target_column] = h2o.H2OFrame(y_test)  # Add target column to the testing frame
#
#         # Define X (features) and y (target)
#         X = [col for col in X_train.columns if col != target_column]
#         y = target_column
#
#         # Track training time
#         start_train_time = time.time()
#         print('Training started using H2O AutoML')
#
#         # Set AutoML hyperparameters
#         max_runtime_secs = int(params.get('max_runtime_secs', 3600))  # Max time in seconds for training
#         max_models = int(params.get('max_models', 20))  # Max models to train
#
#         # Initialize and train AutoML model
#         aml = H2OAutoML(
#             max_models=max_models,
#             max_runtime_secs=max_runtime_secs,
#             seed=42
#         )
#         aml.train(x=X, y=y, training_frame=train_data)
#
#         # End the training time
#         end_train_time = time.time()
#         training_time = end_train_time - start_train_time
#
#         # Get the leaderboard (ranking of models)
#         leaderboard = aml.leaderboard
#         print("Leaderboard:\n", leaderboard)
#
#         # Get the best model (leader model)
#         model = aml.leader
#
#         # Track testing (prediction) time
#         start_test_time = time.time()
#
#         # Predict using the leader model
#         predictions = model.predict(test_data)
#
#         # Convert predictions to a format suitable for sklearn metrics
#         y_pred = predictions.as_data_frame().values.flatten()
#
#         end_test_time = time.time()
#         testing_time = end_test_time - start_test_time
#
#         # Evaluate model performance
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
#         recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
#         f1 = f1_score(y_test, y_pred, average='weighted')
#
#         # Generate plots (optional)
#         plots = generate_visualizations(X_train, X_test, y_train, y_test, model)
#
#         # Prepare the metric results
#         metric_results = {
#             "accuracy": accuracy,
#             "precision": precision,
#             "recall": recall,
#             "f1_score": f1,
#             "training time": training_time,
#             "testing time": testing_time,
#         }
#
#         # Shut down the H2O cluster
#         h2o.shutdown()
#
#         return {
#             "metric_results": remove_none_values(metric_results),
#             "plots": remove_none_values(plots),
#             "model": model,  # Include the best trained model
#             "leaderboard": leaderboard  # Include the leaderboard for all models
#         }
#     else:
#         raise Exception('Target column is required for H2O AutoML classification.')
