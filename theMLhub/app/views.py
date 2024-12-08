from django.shortcuts import render

# Create your views here.
from django.shortcuts import render


def my_view(request):
    return render(request, 'Dashboard.html')


def page_login(request):
    return render(request, 'page-login.html')


def page_register(request):
    return render(request, 'page-register.html')


def chart_peity(request):
    return render(request, 'chart-peity.html')



def chart_sparkline(request):
    return render(request, 'chart-sparkline.html')


def chart_chartist(request):
    return render(request, 'chart-chartist.html')


def chart_chartjs(request):
    return render(request, 'chart-chartjs.html')


def chart_morris(request):
    return render(request, 'chart-morris.html')


def chart_flot(request):
    return render(request, 'chart-flot.html')


# def dashboard(request):
#
#     return render(request, 'Dashboard.html')
def tablePage(request):
    return render(request, 'tablePage.html')


def tableData(request):
    return render(request, 'table-datatable.html')


def app_profile(request):
    return render(request, 'app-profile.html')
