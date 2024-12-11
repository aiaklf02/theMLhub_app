
from django.contrib.auth import authenticate, login
from django.contrib.auth import get_user_model
from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt

from .forms import SignupForm  # Import your SignupForm


from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.contrib.auth.decorators import user_passes_test

from .models import RawDataset


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
            return render(request, 'page-login.html', {'f': 'Invalid username or password' })

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
        form = SignupForm()

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
        # Retrieve file and target column from the request
        uploaded_file = request.FILES.get('file')
        target_column = request.POST.get('target_column')
        utilisateur = request.user  # Assuming the user is authenticated
        print(f"File: {uploaded_file}, Target Column: {target_column}")  # Add logging

        # Validate inputs
        if not uploaded_file:
            return JsonResponse({'error': 'File is required.'}, status=400)

        if not target_column:
            return JsonResponse({'error': 'Target column is required.'}, status=400)

        # Save the file and target column using the RawDataset model
        raw_dataset = RawDataset.objects.create(
            utilisateur=utilisateur,
            file_raw_dataset=uploaded_file,
            selectedTargetColumn=target_column
        )

        return JsonResponse({'success': 'File and target column submitted successfully!'}, status=200)

    return render(request, 'uploadDataFile.html')


def uploadedFiles(request):
    return render(request, 'uploadedFiles.html')

