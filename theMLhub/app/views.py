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
        if not (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.xlsx')):
            return JsonResponse({'error': 'Only CSV and Excel files are supported.'}, status=400)

        try:
            # Read the file to validate its content
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Check if the target column exists
            if target_column not in df.columns:
                return JsonResponse({'error': f'Target column "{target_column}" not found in the uploaded file.'}, status=400)

            # Save the file and dataset details using the RawDataset model
            raw_dataset = RawDataset.objects.create(
                utilisateur=utilisateur,
                file_raw_dataset=uploaded_file,
                TargetColumn=target_column,
                datasetCostumName=custom_name
            )

            # if df:
            #     # PreprocessedDataset
            #
            #     X_train, X_test, y_train, y_test = process_data(df, target_column=target_column)
            #
            #     preprocessedDataset = PreprocessedDataset.objects.create(
            #         raw_dataset=raw_dataset,
            #         # file_preprocessed_data=
            #         preprocessedCostumName=custom_name,
            #     )

            return JsonResponse({'message': 'File uploaded successfully!', 'preprocessing_url': '/preprocessing/'}, status=200)

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



def process_data(df, target_column):
    #first check if the target column is categorical or numerical
    if df[target_column].dtype == 'object':
        # Convert the target column to numerical using label encoding
        df[target_column] = df[target_column].astype('category').cat.codes
    else:
        # The column is already numerical
        pass
    #drop duplicates
    df = df.drop_duplicates()
    #drop missing values if data is large 
    if df.shape[0] > 1000:
        df = df.dropna()
    else:
        #fill missing values with the mean
        df = df.fillna(df.mean())
    #detect outliers
    if df.shape[0] > 1000:
        #detect outliers using z-score
        from scipy import stats
        import numpy as np
        z = np.abs(stats.zscore(df))
        df = df[(z < 3).all(axis=1)]
    else:
        #detect outliers using IQR
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df <  (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
     #split the data into features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    #balance the dataset using smote if the data is imbalanced
    if df[target_column].value_counts().min() < 0.6 * df[target_column].value_counts().max():

        from imblearn.over_sampling import SMOTE
        smote = SMOTE()
        X, y = smote.fit_resample(df.drop(target_column, axis=1), df[target_column])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test



   
    
    
        

