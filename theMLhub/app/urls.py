# my_app/urls.py
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.my_view, name='index'),  # The root URL maps to my_view
    path('page-login', views.page_login, name='page-login'),  # The root URL maps to my_view
    path('page-register', views.page_register, name='page-register'),  # The root URL maps to my_view
    path('tablePage', views.tablePage, name='tablePage'),  # The root URL maps to my_view
    path('tableData', views.tableData, name='tableData'),  # The root URL maps to my_view
    path('app-profile', views.app_profile, name='app_profile'),  # The root URL maps to my_view
    path('uploadDataFile', views.uploadDataFile, name='uploadDataFile'),  # The root URL maps to my_view

    path('uploadedFiles', views.uploadedFiles, name='uploadedFiles'),  # The root URL maps to my_view
    path('classification', views.classification, name='classification'),  # The root URL maps to my_view
    path('regression', views.regression, name='regression'),  # The root URL maps to my_view
    path('clustering', views.clustering, name='clustering'),  # The root URL maps to my_view

    path('chart-flot', views.chart_flot, name='chart-flot'),  # The root URL maps to my_view
    path('chart-morris', views.chart_morris, name='chart-morris'),  # The root URL maps to my_view
    path('chart-chartjs', views.chart_chartjs, name='chart-chartjs'),  # The root URL maps to my_view
    path('chart-chartist', views.chart_chartist, name='chart-chartist'),  # The root URL maps to my_view
    path('chart-sparkline', views.chart_sparkline, name='chart-sparkline'),  # The root URL maps to my_view
    path('chart-peity', views.chart_peity, name='chart-peity'),  # The root URL maps to my_view
    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
