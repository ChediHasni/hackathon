"""
ZoneScore API URL Configuration
"""
from django.urls import path
from api import views

urlpatterns = [
    path('health/',   views.HealthView.as_view(),   name='api-health'),
    path('criteria/', views.CriteriaView.as_view(), name='api-criteria'),
    path('score/',    views.ScoreView.as_view(),    name='api-score'),
]
