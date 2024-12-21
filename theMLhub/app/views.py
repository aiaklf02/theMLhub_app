from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from .models import RawDataset, PreprocessedDataset, DataVisualization
from django.http import HttpResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth import get_user_model
from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import json
from .forms import SignupForm  # Import your SignupForm
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.contrib.auth.decorators import user_passes_test
from django.core.files.base import File
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from .models import RawDataset, PreprocessedDataset, DataVisualization




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
                    target_column = target_column.replace('"','')
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
            if not os.path.exists(file_path):
                return JsonResponse({'error': f'File path {file_path} does not exist.'}, status=500)
            # Process the data
            #add file path to the preprocessed dataset
            preprocessed_dataset = PreprocessedDataset.objects.create(
                raw_dataset=raw_dataset,
                preprocessedCostumName=custom_name+'_preprocessed',
            )
          

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



from .ML_Models import *

# A dictionary mapping model names to their corresponding functions
# MODEL_FUNCTIONS = {
MODEL_FUNCTIONS = {
    "Linear Regression": train_linear_regression,
    "Regression LightGBM": train_regression_LightGBM,
    "Decision Trees": decisionTreeCart,
    "Random Forest": train_random_forest,
    "K-Nearest Neighbors": train_knn,
    "Support Vector Machines (SVR)": train_svr,
    "XGBoost": train_xgboost,
    "Reseau Neuron": train_reseau_neuron,

    "K-Means": KMeansClustering,

    "Classification LightGBM": train_classification_LightGBM,
    "Logistic Regression": train_logistic_regression,
    "Naive bayes": train_classification_naiveBayes,
    "Decision Tree": train_classification_cart_decision_tree,
    "Random Forests": train_classification_random_forest,
    "K Nearest Neighbors": train_classification_knn,
    "Support Vector Machines (SVC)": train_classification_svc,
    "XG Boost": train_classification_xgboost,
    "Reseau Neurons": train_classification_reseau_neuron,
    # Add more models here as needed

}


@login_required_custom
@csrf_exempt
def train_model_view(request, model_name, processed_file_id, supervised):
    if request.method == "POST":
        # Extract 'params' from the POST body
        params = request.POST.get("params")  # Use this if the data is form-encoded
        params_dict = json.loads(params)

        # Fetch the PreprocessedDataset object
        preprocessed_dataset = PreprocessedDataset.objects.get(id=processed_file_id)

        # Fetch the target column from the associated RawDataset
        target_column = preprocessed_dataset.raw_dataset.TargetColumn

        if supervised == "supervised":
            processedData = preprocessed_dataset.process_data(target_column)
        else:
            target_column = None
            processedData = preprocessed_dataset.process_data_unsupervised()

        # Check if the selected model exists in the mapping
        model_function = MODEL_FUNCTIONS.get(model_name)

        context = {
            "message": "Couldnt Start traning , Invalid model or dataset !",
            "result": 'no result to show',
            "status": 'failed',
            "modelName": model_name,
            "dataCostumName": preprocessed_dataset.raw_dataset.datasetCostumName
        }

        try:
            # Execute the associated function, passing the file path and target column
            result = model_function(processedData,params=params_dict, target_column=target_column)

            context = {
                "message": f"Model trained successfully",
                "result": result,
                "status": 'success',
                "modelName": model_name,
                "dataCostumName": preprocessed_dataset.raw_dataset.datasetCostumName
            }

        except Exception as e:
            context = {
                "message": f"Error during training",
                "result": f'{str(e)}',
                "status":'failed',
                "modelName": model_name,
                "dataCostumName": preprocessed_dataset.raw_dataset.datasetCostumName
            }
            raise e

        return render(request, 'train_result.html', context)


def visualize_data(request, datatype,dataset_id):
    """
    Vue pour afficher les visualisations associées à un jeu de données brut ou prétraité.
    """
    if datatype == 'raw':
        # Recherche dans RawDataset
        dataset = RawDataset.objects.filter(id=dataset_id).first()
        data_visualizations = DataVisualization.objects.filter(dataset=dataset, dataset_processed__isnull=True)

        if not data_visualizations.exists():
            dataset.generate_visualizations()
            data_visualizations = DataVisualization.objects.filter(dataset=dataset, dataset_processed__isnull=True)

        return render(request, 'visualisationData.html', {
            'data_visualizations': data_visualizations,
            'dataset': dataset,
        })

    else:
        dataset = PreprocessedDataset.objects.filter(id=dataset_id).first()
        data_visualizations = DataVisualization.objects.filter(dataset_processed=dataset)

        if not data_visualizations.exists():
            dataset.generate_visualizations()
            data_visualizations = DataVisualization.objects.filter(dataset_processed=dataset)

        return render(request, 'visualisationData.html', {
            'data_visualizations': data_visualizations,
            'dataset': dataset,
        })

