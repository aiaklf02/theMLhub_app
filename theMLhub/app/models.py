import os

from django.contrib.auth.models import AbstractUser
from django.db import models


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
    selectedTargetColumn = models.CharField(max_length=255, null=False, default='target')


class PreprocessedDataset(models.Model):
    raw_dataset = models.ForeignKey(RawDataset, on_delete=models.CASCADE)
    file_preprocessed_data = models.FileField(upload_to='processed_datasets/')
    processed_at = models.DateTimeField(auto_now_add=True)


class AiModel(models.Model):
    name = models.CharField(max_length=100)
    model_params = models.JSONField()  # For flexibility


class Result(models.Model):
    ai_model = models.ForeignKey(AiModel, on_delete=models.CASCADE)  # Many-to-one with AiModel
    preprocessed_dataset = models.ForeignKey(PreprocessedDataset,
                                             on_delete=models.CASCADE)  # Many-to-one with PreprocessedDataset
    accuracy = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    mse = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)  # Track when the result was created


class DataGraph(models.Model):
    Graphname = models.CharField(max_length=100, default='Graph Name')
    result = models.ForeignKey(Result, on_delete=models.CASCADE)
    graph_file = models.FileField(upload_to='graphs/', null=True, blank=True)
