import base64
import io
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report




def generate_visualizations(X_train, X_test, y_train, y_test, model, fpr=None, tpr=None, classification_rep=None):
    """Generate various visualizations based on the model's predictions and dataset.

        Arguments:
        - X_train: Training features (DataFrame)
        - X_test: Testing features (DataFrame)
        - y_train: Actual target values for training (array or list)
        - y_test: Actual target values for testing (array or list)
        - model: Trained model (object)
        - fpr: False positive rate for ROC curve (array or list, optional)
        - tpr: True positive rate for ROC curve (array or list, optional)
        - classification_rep: Classification report (string, optional)
        """
    plots = {}

    # Generate and store the ROC curve plot if fpr and tpr are provided
    if fpr is not None and tpr is not None:
        plots['roc curve'] = generate_roc_curve_plot(fpr, tpr)

    # Generate and store the classification report plot if provided
    if classification_rep is not None:
        plots['classification report'] = generate_classification_report_plot2(classification_rep)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate residuals
    residuals = y_test - predictions

    # Generate and store the residual plot
    plots['residual plot'] = generate_residual_plot(predictions, residuals)

    # Generate and store the prediction vs actual plot
    plots['prediction vs actual'] = generate_prediction_vs_actual_plot(y_test, predictions)

    # Generate and store the histogram of residuals plot
    plots['residual histogram'] = generate_histogram_of_residuals(residuals)

    # Generate and store the coefficients plot
    plots['feature coefficients'] = generate_coefficients_plot(X_train, model)

    return plots



def save_plot_to_base64():
    """Save the current matplotlib plot to a base64-encoded string."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()
    return image_base64


def generate_roc_curve_plot(fpr, tpr):
    """Generate and return the ROC curve plot.

        Arguments:
        - fpr: False positive rate values (array or list)
        - tpr: True positive rate values (array or list)
        """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label="ROC Curve")
    plt.plot([0, 1], [0, 1], color='gray', linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    return save_plot_to_base64()




# Function to generate Classification Report plot
def generate_classification_report_plot(y_test, predictions):
    report = classification_report(y_test, predictions, output_dict=True)
    classification_rep = str(report)

    plt.figure(figsize=(8, 6))
    plt.text(0.1, 1.1, f"Classification Report:\n{classification_rep}", fontsize=12)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    classification_report_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return classification_report_plot

def generate_classification_report_plot2(classification_rep):
    """Generate and return the classification report plot.

        Arguments:
        - classification_rep: The classification report (string)
        """
    plt.figure(figsize=(8, 6))
    plt.text(0.1, 1.1, f"Classification Report:\n{classification_rep}", fontsize=12)
    plt.axis('off')
    return save_plot_to_base64()


def generate_residual_plot(predictions, residuals):
    """Generate and return the residual plot.

        Arguments:
        - predictions: Predicted values (array or list)
        - residuals: Residuals (array or list)
        """
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    return save_plot_to_base64()


def generate_prediction_vs_actual_plot(y_test, predictions):
    """Generate and return the prediction vs actual plot.

        Arguments:
        - y_test: Actual target values (array or list)
        - predictions: Predicted values (array or list)
        """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
    plt.title("Prediction vs Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    return save_plot_to_base64()


def generate_histogram_of_residuals(residuals):
    """Generate and return the histogram of residuals plot.

        Arguments:
        - residuals: Residuals (array or list)
        """
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    return save_plot_to_base64()


def generate_coefficients_plot(X_train, model):
    try:
        """Generate and return the feature coefficients plot.
    
            Arguments:
            - X_train: Training data features (DataFrame)
            - model: Trained model (object)
            """
        plt.figure(figsize=(10, 6))
        plt.bar(X_train.columns, model.coef_, alpha=0.7)
        plt.title("Feature Coefficients")
        plt.xlabel("Features")
        plt.ylabel("Coefficient Value")
        plt.xticks(rotation=45, ha="right")
        return save_plot_to_base64()
    except:
        pass

def generate_target_distribution_plot(data, target_column):
    """Generate and return the target distribution plot.

        Arguments:
        - data: DataFrame containing the dataset
        - target_column: Target variable column name (string)
        """
    plt.figure(figsize=(8, 6))
    sns.histplot(data[target_column], kde=True, color='blue')
    plt.title(f"Distribution of {target_column}")
    return save_plot_to_base64()


def generate_correlation_heatmap(data):
    """Generate and return the correlation heatmap plot.

        Arguments:
        - data: DataFrame containing the dataset
        """
    plt.figure(figsize=(12, 8))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title("Correlation Heatmap")
    return save_plot_to_base64()


def generate_bivariate_scatter_plot(data, target_column, feature):
    """Generate and return a bivariate scatter plot for target vs feature.

        Arguments:
        - data: DataFrame containing the dataset
        - target_column: Target variable column name (string)
        - feature: Feature column name for scatter plot (string)
        """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[feature], y=data[target_column])
    plt.title(f"{target_column} vs {feature}")
    plt.xlabel(feature)
    plt.ylabel(target_column)
    return save_plot_to_base64()


def generate_pairplot(data, numerical_features, target_column):
    """Generate and return the pairplot for selected numerical features.

        Arguments:
        - data: DataFrame containing the dataset
        - numerical_features: List of numerical features (list of strings)
        - target_column: Target variable column name (string)
        """
    plt.figure(figsize=(10, 10))
    sampled_features = numerical_features[:5] + [target_column]
    sns.pairplot(data[sampled_features])
    plt.title("Pairplot of Features")
    return save_plot_to_base64()


def generate_boxplot(data, target_column, feature):
    """Generate and return the boxplot for categorical features vs target.

        Arguments:
        - data: DataFrame containing the dataset
        - target_column: Target variable column name (string)
        - feature: Categorical feature column name (string)
        """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[feature], y=data[target_column])
    plt.title(f"{target_column} vs {feature}")
    plt.xlabel(feature)
    plt.ylabel(target_column)
    return save_plot_to_base64()


# Function to generate Confusion Matrix plot
def generate_confusion_matrix_plot(y_test, predictions, model):
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    confusion_matrix_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return confusion_matrix_plot
