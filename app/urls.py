from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Головна сторінка
    path('cluster/', views.cluster, name='cluster'),  # Кластеризація
    path('info/', views.info, name='info'),  # Інформаційна сторінка
    path('teach/', views.teach, name='teach'),  # Навчальна сторінка
]
