from django.contrib.auth import authenticate, login
from django.contrib.auth import get_user_model
from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
import pandas as pd

from .forms import SignupForm  # Import your SignupForm

from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.contrib.auth.decorators import user_passes_test
from django.core.files.base import File
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

from .models import RawDataset, PreprocessedDataset


def login_required_custom(view_func):
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return HttpResponseRedirect(reverse('page-login'))  # Redirect to login page
        return view_func(request, *args, **kwargs)

    return _wrapped_view


# @login_required_custom
def my_view(request):
    return render(request, 'Dashboard.html')


def page_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')  # Redirect to a success page (e.g., home)
        else:
            message = 'Invalid username or password'
            return render(request, 'page-login.html', {'f': 'Invalid username or password'})

    return render(request, 'page-login.html')


def page_register(request):
    if request.method == 'POST':
        form = SignupForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, 'Account created successfully! You can now log in.')
            return redirect('page-login')
        else:
            print(form.errors)  # Add this to log the errors

            messages.error(request, 'There were errors in your form. Please check.')
    else:
           form = SignupForm(initial={
            'username': '',
            'email': '',
            'profile_picture_path': None,
            'password1': '',
            'password2': '',
            'country': '',
            'full_name': '',
           
        })

    return render(request, 'page-register.html', {'form': form})


# @login_required_custom
def chart_peity(request):
    return render(request, 'chart-peity.html')


# @login_required_custom
def chart_sparkline(request):
    return render(request, 'chart-sparkline.html')


# @login_required_custom
def chart_chartist(request):
    return render(request, 'chart-chartist.html')


# @login_required_custom
def chart_chartjs(request):
    return render(request, 'chart-chartjs.html')


# @login_required_custom
def chart_morris(request):
    return render(request, 'chart-morris.html')


# @login_required_custom
def chart_flot(request):
    return render(request, 'chart-flot.html')


# @login_required_custom
def tablePage(request):
    return render(request, 'tablePage.html')


# @login_required_custom
def tableData(request):
    return render(request, 'table-datatable.html')


# @login_required_custom
def app_profile(request):
    return render(request, 'app-profile.html')

import chardet 

import os
@csrf_exempt    

def uploadDataFile(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        target_column = request.POST.get('target_column')  # Optional for unsupervised
        custom_name = request.POST.get('custom_name')
        utilisateur = request.user  # Assuming the user is authenticated

        # Validate inputs
        if not uploaded_file:
            return JsonResponse({'error': 'File is required.'}, status=400)
        if not custom_name:
            return JsonResponse({'error': 'Dataset custom name is required.'}, status=400)

        # Validate file type (accepting both CSV and Excel)
        if not (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls')):
            return JsonResponse({'error': 'Only CSV and Excel files are supported.'}, status=400)

        try:
            # Handle CSV and Excel files
            if uploaded_file.name.endswith('.csv'):
                # Detect file encoding
                raw_data = uploaded_file.read(1000)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'ISO-8859-1'  # Default to 'ISO-8859-1' if detection fails
                uploaded_file.seek(0)  # Reset file pointer after reading
                df = pd.read_csv(uploaded_file, encoding=encoding)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, engine='openpyxl')  # For newer Excel files
            elif uploaded_file.name.endswith('.xls'):
                df = pd.read_excel(uploaded_file, engine='xlrd')  # For older Excel files

            # Check if the dataset is empty
            if df.empty:
                return JsonResponse({'error': 'The dataset is empty.'}, status=400)

            # For supervised learning, validate target column
            if target_column:
                if target_column not in df.columns:
                    return JsonResponse({'error': f'Target column "{target_column}" not found in the uploaded file.'}, status=400)

            # Save the file and dataset details using the RawDataset model
            raw_dataset = RawDataset.objects.create(
                utilisateur=utilisateur,
                file_raw_dataset=uploaded_file,
                TargetColumn=target_column if target_column else None,
                datasetCostumName=custom_name
            )

            file_path = raw_dataset.file_raw_dataset.path
            print(f"File path: {file_path}")
            if not os.path.exists(file_path):
                return JsonResponse({'error': f'File path {file_path} does not exist.'}, status=500)

            # Process the data
            preprocessed_dataset = PreprocessedDataset.objects.create(raw_dataset=raw_dataset)
            if target_column:
                # Supervised workflow
                try:
                    X_train, X_test, y_train, y_test = preprocessed_dataset.process_data(target_column)
                except Exception as e:
                    print(f"Error during supervised data processing: {e}")
                    return JsonResponse({'error': 'An error occurred during data preprocessing.'}, status=500)
            else:
                # Unsupervised workflow
                try:
                    processed_data = preprocessed_dataset.process_data_unsupervised()
                except Exception as e:
                    print(f"Error during unsupervised data processing: {e}")
                    return JsonResponse({'error': 'An error occurred during unsupervised data processing.'}, status=500)

            # Return a success response
            return JsonResponse({'message': 'File uploaded and processed successfully!',
                                 'preprocessing_url': '/preprocessing/'}, status=200)

        except Exception as e:
            # Log the exception for debugging
            print(f"Error processing file: {e}")
            return JsonResponse({'error': f'An error occurred while processing the file: {str(e)}'}, status=500)

    return render(request, 'uploadDataFile.html')


def getUploadedDatasets(request):
    uploadedfiles = RawDataset.objects.filter(utilisateur=request.user)
    processeddatasets = PreprocessedDataset.objects.filter(raw_dataset__utilisateur=request.user)
    return uploadedfiles, processeddatasets


@login_required_custom
def uploadedFiles(request):
    uploadedfiles, processeddatasets = getUploadedDatasets(request)
    form = {'uploadedfiles': uploadedfiles,'processeddatasets':processeddatasets}
    return render(request, 'uploadedFiles.html', form)

@login_required_custom
def classification(request):
    uploadedfiles, processeddatasets = getUploadedDatasets(request)
    form = {'uploadedfiles': uploadedfiles,'processeddatasets':processeddatasets}

    return render(request, 'classification.html', form)

@login_required_custom
def regression(request):
    uploadedfiles, processeddatasets = getUploadedDatasets(request)
    form = {'uploadedfiles': uploadedfiles,'processeddatasets':processeddatasets}

    return render(request, 'regression.html', form)

@login_required_custom
def clustering(request):
    uploadedfiles, processeddatasets = getUploadedDatasets(request)
    form = {'uploadedfiles': uploadedfiles,'processeddatasets':processeddatasets}

    return render(request, 'clustering.html', form)



from .ML_Models import train_linear_regression, generate_visualizations, train_logistic_regression

# A dictionary mapping model names to their corresponding functions
MODEL_FUNCTIONS = {
    "Linear Regression": train_linear_regression,
    "Logistic Regression": train_logistic_regression,
    # Add more models here as needed
}

@login_required_custom
def train_model_view(request):
    if request.method == "POST":

        selected_model = request.POST.get('SelectedModel')
        selected_dataset_id = request.POST.get('selectedDataset')

        # Ensure both model and dataset are provided
        if not selected_model or not selected_dataset_id:
            return JsonResponse({"error": "Model and dataset must be selected"}, status=400)

        # Fetch the PreprocessedDataset object
        preprocessed_dataset = PreprocessedDataset.objects.get(id=selected_dataset_id)

        # Fetch the target column from the associated RawDataset
        target_column = preprocessed_dataset.raw_dataset.TargetColumn

        if target_column:
            processedData = preprocessed_dataset.process_data(target_column)
        else:
            target_column = None
            processedData = preprocessed_dataset.process_data_unsupervised()

        # Check if the selected model exists in the mapping
        model_function = MODEL_FUNCTIONS.get(selected_model)

        try:
            # Execute the associated function, passing the file path and target column
            result = model_function(processedData, target_column)

            return JsonResponse({
                "message": f"Model '{selected_model}' trained successfully",
                "result": result,  # Include function output
            })
        except Exception as e:
            return JsonResponse({"error": f"Error during training: {str(e)}"}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=400)


@login_required_custom
def visualize_data_view(request):
    selected_dataset_id = request.POST.get('selectedDataset')
    target_column = request.POST.get('TargetColumn')

    # Fetch the dataset
    try:
        preprocessed_data = PreprocessedDataset.objects.get(id=selected_dataset_id)
        file_path = preprocessed_data.file_preprocessed_data.path
    except PreprocessedDataset.DoesNotExist:
        return JsonResponse({"error": f"Dataset with ID {selected_dataset_id} does not exist"}, status=404)

    # Generate visualizations
    visualizations = generate_visualizations(file_path, target_column)
    if "error" in visualizations:
        return JsonResponse({"error": visualizations["error"]}, status=400)

    return JsonResponse({"visualizations": visualizations})