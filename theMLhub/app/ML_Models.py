import io
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, classification_report, \
    roc_auc_score, roc_curve, matthews_corrcoef, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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


def train_linear_regression(preprocesseddata, target_column=None):

    if target_column:
        X_train, X_test, y_train, y_test = preprocesseddata

        # Track training time
        start_train_time = time.time()
        #
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
            "training_time": training_time,
            "testing_time": testing_time,
        }


        obj = {
            "metric_results": metric_results,
            "plots": plots,
            "model": model,  # Include the trained model
        }

        return obj

    else:
        raise Exception('linear regression require target Column !')

def train_logistic_regression(preprocesseddata, target_column=None):

    if target_column:
        X_train, X_test, y_train, y_test = preprocesseddata

        # Track training time
        start_train_time = time.time()
        #
        # Train model
        model = LogisticRegression()
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

        plots['classification_report'] = generate_classification_report_plot(y_test, predictions)
        plots['confusion_matrix'] = generate_confusion_matrix_plot(y_test, predictions, model)

        metric_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "training_time": training_time,
            "testing_time": testing_time,
        }

        obj = {
            "metric_results": metric_results,
            "plots": plots,
            "model": model,  # Include the trained model

        }

        return obj
    else:
        raise Exception('Target column is required for Logistic Regression.')



# draft

# # Evaluate the model
# metric_results = {
#     "mse": mean_squared_error(y_test, predictions),
#     "mae": mean_absolute_error(y_test, predictions),
#     "r2": r2_score(y_test, predictions),
#     "auc": auc_score,
#     "mcc": mcc,
#     "accuracy": classification_rep['accuracy'],
#     "f1_score": classification_rep['weighted avg']['f1-score'],
#     "precision": classification_rep['weighted avg']['precision'],
#     "recall": classification_rep['weighted avg']['recall'],
#     "training_time": training_time,
#     "testing_time": testing_time,
# }

# Save the trained model file
# model_file_path = f"trained_models/{selected_model_name}_{preprocessed_data.id}.joblib"
# dump(model, model_file_path)

# ai_model =
# Save results to the database
# with open(model_file_path, 'rb') as model_file:
#     result = Result(
#         ai_model=ai_model,
#         preprocessed_dataset=preprocessed_data,
#         accuracy=accuracy,
#         f1_score=f1,
#         mse=mse,
#         mae=mae
#     )
#     result.trained_model_file.save(f"{selected_model_name}_{preprocessed_data.id}.joblib", File(model_file))
#     result.save()

#  end draft