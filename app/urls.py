from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('cluster/', views.cluster, name='cluster'),
    path('info/', views.info, name='info'),
    path('teach/', views.teach, name='teach'),
]
