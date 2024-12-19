import os

from django.contrib.auth.models import AbstractUser
from django.db import models
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from django.core.files.storage import default_storage


class Utilisateur(AbstractUser):
    profile_picture_path = models.FileField(upload_to='Profile_pictures/', null=True, blank=True)
    country = models.CharField(max_length=100, null=True, blank=True)
    full_name = models.CharField(max_length=150, null=True, blank=True)
    STATUS_CHOICES = [
        ('Student', 'Student'),
        ('Professor', 'Professor'),
        ('Employee', 'Employee'),
    ]
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, null=True, blank=True)


class RawDataset(models.Model):
    utilisateur = models.ForeignKey(Utilisateur, on_delete=models.CASCADE)
    file_raw_dataset = models.FileField(upload_to='raw_datasets/')
    datasetCostumName = models.CharField(max_length=100, default='DataSetFile' ,null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    TargetColumn = models.CharField(max_length=255, null=True,blank=True,default='target')
    def __str__(self):
        return self.datasetCostumName
    #load the data from the file


from django.http import JsonResponse

class PreprocessedDataset(models.Model):
    raw_dataset = models.ForeignKey(RawDataset, on_delete=models.CASCADE)
    file_preprocessed_data = models.FileField(upload_to='processed_datasets/')
    processed_at = models.DateTimeField(auto_now_add=True)
    preprocessedCostumName = models.CharField(max_length=100, default='PreprocessedDataSetFile', null=True, blank=True)

    def process_data(self, target_column):
        # Charger le dataset brut
        raw_file_path = self.raw_dataset.file_raw_dataset.path
        if not os.path.exists(raw_file_path):
            raise FileNotFoundError(f"The file {raw_file_path} does not exist.")
        
        # Vérifier l'extension du fichier pour choisir la méthode de lecture appropriée
        if raw_file_path.endswith('.csv'):
            df = pd.read_csv(raw_file_path)
        elif raw_file_path.endswith('.xls') or raw_file_path.endswith('.xlsx'):
            df = pd.read_excel(raw_file_path, engine='openpyxl' if raw_file_path.endswith('.xlsx') else 'xlrd')
        else:
            raise ValueError(f"Unsupported file type: {raw_file_path}")

        # Vérifications initiales
        if target_column not in df.columns:
            raise ValueError(f"The target column '{target_column}' is not found in the dataset.")
        
        # Convertir les colonnes cibles catégoriques
        if df[target_column].dtype == 'object':
            df[target_column] = df[target_column].astype('category').cat.codes

        # Enlever les doublons
        df = df.drop_duplicates()

        # Gérer les valeurs manquantes - Traitement séparé pour les colonnes numériques et catégoriques
        for col in df.columns:
            if df[col].dtype == 'object':  # Si la colonne est catégorique
                df[col] = df[col].fillna(df[col].mode()[0])  # Remplacer les NaN par la valeur la plus fréquente
            else:  # Si la colonne est numérique
                df[col] = df[col].fillna(df[col].median())  # Remplacer les NaN par la médiane

        # Séparer X et y
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        if pd.api.types.is_numeric_dtype(y):
            pass
        else:
              # Équilibrage des données avec SMOTE
            if y.value_counts().min() < 0.6 * y.value_counts().max():
                smote = SMOTE()
                X, y = smote.fit_resample(X, y)
            
        # Séparer les ensembles de données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalisation (seulement pour les colonnes numériques)
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns  # Sélection des colonnes numériques
        sc = StandardScaler()
        X_train[numeric_cols] = sc.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = sc.transform(X_test[numeric_cols])

        # Sauvegarde des ensembles dans un CSV
        train_data = pd.DataFrame(X_train, columns=X.columns)
        train_data['target'] = y_train
        train_data['set'] = 'train'

        test_data = pd.DataFrame(X_test, columns=X.columns)
        test_data['target'] = y_test
        test_data['set'] = 'test'

        processed_data = pd.concat([train_data, test_data], ignore_index=True)

        processed_datasets_dir = os.path.join('media', 'processed_datasets')
        os.makedirs(processed_datasets_dir, exist_ok=True)
        processed_file_path = os.path.join(processed_datasets_dir, f'{self.preprocessedCostumName}.csv')

        processed_data.to_csv(processed_file_path, index=False)

        with open(processed_file_path, 'rb') as f:
            self.file_preprocessed_data.save(f'{self.preprocessedCostumName}.csv', f)
     
        return X_train, X_test, y_train, y_test

    def process_data_unsupervised(self):
        # Load the raw dataset
        raw_file_path = self.raw_dataset.file_raw_dataset.path
        if not os.path.exists(raw_file_path):
            raise FileNotFoundError(f"The file {raw_file_path} does not exist.")
        
        # Read the dataset based on file type++-
        if raw_file_path.endswith('.csv'):
            df = pd.read_csv(raw_file_path)
        elif raw_file_path.endswith('.xls') or raw_file_path.endswith('.xlsx'):
            df = pd.read_excel(raw_file_path, engine='openpyxl' if raw_file_path.endswith('.xlsx') else 'xlrd')
        else:
            raise ValueError(f"Unsupported file type: {raw_file_path}")

        # Drop duplicates
        df = df.drop_duplicates()

        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':  # For categorical columns
                df[col] = df[col].fillna(df[col].mode()[0])
            else:  # For numerical columns
                df[col] = df[col].fillna(df[col].median())

        # Encode categorical variables
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes

        # Normalize numerical columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Save the processed dataset
        processed_datasets_dir = os.path.join('media', 'processed_datasets')
        os.makedirs(processed_datasets_dir, exist_ok=True)
        processed_file_path = os.path.join(processed_datasets_dir, f'{self.preprocessedCostumName}_unsupervised.csv')

        df.to_csv(processed_file_path, index=False)

        with open(processed_file_path, 'rb') as f:
            self.file_preprocessed_data.save(f'{self.preprocessedCostumName}_unsupervised.csv', f)

        return df


class AiModel(models.Model):
    name = models.CharField(max_length=100)
    model_params = models.JSONField()  # For flexibility


class Result(models.Model):
    ai_model = models.ForeignKey(AiModel, on_delete=models.CASCADE)  # Many-to-one with AiModel
    preprocessed_dataset = models.ForeignKey(PreprocessedDataset, on_delete=models.CASCADE)  # Many-to-one with PreprocessedDataset
    trained_model_file = models.FileField(upload_to='trained_models/', null=True, blank=True)

    training_time = models.FloatField(null=True, blank=True)
    testing_time = models.FloatField(null=True, blank=True)

    r2_score = models.FloatField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)

    mse = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)
    MCC = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)  # Track when the result was created


class DataGraph(models.Model):
    Graphname = models.CharField(max_length=100, default='Graph Name')
    result = models.ForeignKey(Result, on_delete=models.CASCADE)
    graph_file = models.FileField(upload_to='graphs/', null=True, blank=True)

