from django.urls import path
from . import views

urlpatterns = [
    path('home', views.home, name='home'),
    # Add more URL patterns if needed
]