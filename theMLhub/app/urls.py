# my_app/urls.py
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.my_view, name='index'),
    path('home', views.home, name='home'),
    path('page-login', views.page_login, name='page-login'),
    path('Logout', views.logout, name='Logout'),
    path('page-register', views.page_register, name='page-register'),
    path('tablePage', views.tablePage, name='tablePage'),
    path('tableData', views.tableData, name='tableData'),
    path('app-profile', views.app_profile, name='app_profile'),
    path('uploadDataFile', views.uploadDataFile, name='uploadDataFile'),
    path('uploadedFiles', views.uploadedFiles, name='uploadedFiles'),
    path('classification', views.classification, name='classification'),
    path('regression', views.regression, name='regression'),
    path('clustering', views.clustering, name='clustering'),
    path('Start_training', views.train_model_view, name='start_traning'),
    path('train_model/<path:model_name>/<str:processed_file_id>/<str:supervised>/', views.train_model_view, name='train_model_view'),
    path('chart-flot', views.chart_flot, name='chart-flot'),
    path('chart-morris', views.chart_morris, name='chart-morris'),
    path('chart-chartjs', views.chart_chartjs, name='chart-chartjs'),
    path('chart-chartist', views.chart_chartist, name='chart-chartist'),
    path('chart-sparkline', views.chart_sparkline, name='chart-sparkline'),
    path('chart-peity', views.chart_peity, name='chart-peity'),
    path('All-Results', views.Results, name='All-Results'),
    path('visualise-data/<str:datatype>/<int:dataset_id>/', views.visualize_data, name='visualize_data'),
    path('visualise-result/<int:resultID>/', views.visualize_result, name='visualize_result'),
    path('download_report/<int:resultID>/', views.download_report, name='download_report'),
    path('download_excel/<int:resultID>/', views.download_excel, name='download_excel'),
    path('download_preproccessed_data/<int:prepdataID>/', views.downloadPreproccesseddata, name='downloadPreproccesseddata'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)