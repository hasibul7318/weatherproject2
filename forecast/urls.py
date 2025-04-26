# D:\machinelearng03\weatherproject2\forecast\urls.py
from django.urls import path
from .views import weather_view  # Ensure this line is correct

urlpatterns = [
    path('', weather_view, name='weather_view'),
]
