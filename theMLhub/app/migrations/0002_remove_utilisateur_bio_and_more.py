# Generated by Django 5.1.4 on 2024-12-10 13:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='utilisateur',
            name='bio',
        ),
        migrations.RemoveField(
            model_name='utilisateur',
            name='mobile_number',
        ),
        migrations.AddField(
            model_name='utilisateur',
            name='status',
            field=models.CharField(blank=True, choices=[('Student', 'Student'), ('Professor', 'Professor'), ('Employee', 'Employee')], max_length=50, null=True),
        ),
    ]
