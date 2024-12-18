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
        target_column = request.POST.get('target_column')
        custom_name = request.POST.get('custom_name')
        utilisateur = request.user  # Assuming the user is authenticated

        # Validate inputs
        if not uploaded_file:
            return JsonResponse({'error': 'File is required.'}, status=400)
        if not target_column:
            return JsonResponse({'error': 'Target column is required.'}, status=400)
        if not custom_name:
            return JsonResponse({'error': 'Dataset custom name is required.'}, status=400)

        # Validate file type (accepting both CSV and Excel)
        if not (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls')):
            return JsonResponse({'error': 'Only CSV and Excel files are supported.'}, status=400)

        try:
            # Handle CSV and Excel files
            if uploaded_file.name.endswith('.csv'):
                # Detect file encoding (use chardet to auto-detect encoding)
                raw_data = uploaded_file.read(1000)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'ISO-8859-1'  # Default to 'ISO-8859-1' if detection fails
                uploaded_file.seek(0)  # Reset file pointer after reading
                df = pd.read_csv(uploaded_file, encoding=encoding, errors='ignore')  # Handle CSV encoding
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, engine='openpyxl')  # For newer Excel files
            elif uploaded_file.name.endswith('.xls'):
                df = pd.read_excel(uploaded_file, engine='xlrd')  # For older Excel files

            # Check if the target column exists
            if target_column not in df.columns:
                return JsonResponse({'error': f'Target column "{target_column}" not found in the uploaded file.'}, status=400)

            if df.empty:
                return JsonResponse({'error': 'The dataset is empty.'}, status=400)

            # Save the file and dataset details using the RawDataset model
            raw_dataset = RawDataset.objects.create(
                utilisateur=utilisateur,
                file_raw_dataset=uploaded_file,
                TargetColumn=target_column,
                datasetCostumName=custom_name
            )

            file_path = raw_dataset.file_raw_dataset.path
            print(f"File path: {file_path}")
            if not os.path.exists(file_path):
                return JsonResponse({'error': f'File path {file_path} does not exist.'}, status=500)

            # Process the data
            preprocessed_dataset = PreprocessedDataset.objects.create(raw_dataset=raw_dataset)
            try:
                X_train, X_test, y_train, y_test = preprocessed_dataset.process_data(target_column)
            except Exception as e:
                print(f"Error during data processing: {e}")
                return JsonResponse({'error': 'An error occurred during data preprocessing.'}, status=500)

            # You can also return a URL for the preprocessed dataset if needed
            return JsonResponse({'message': 'File uploaded and processed successfully!', 
                                  'preprocessing_url': '/preprocessing/'}, status=200)

        except Exception as e:
            # Log the exception for debugging
            print(f"Error processing file: {e}")
            return JsonResponse({'error': f'An error occurred while processing the file: {str(e)}'}, status=500)

    return render(request, 'uploadDataFile.html')



@login_required_custom
def uploadedFiles(request):
    uploadedfilesbyme = RawDataset.objects.filter(utilisateur=request.user)
    print(f'My uploaded files: {uploadedfilesbyme}')
    return render(request, 'uploadedFiles.html', {'files': uploadedfilesbyme})





   
    
    
        

