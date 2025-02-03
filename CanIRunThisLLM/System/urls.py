from django.urls import path
from .views import home, upload_system_info

urlpatterns = [
    path('', home, name='home'),
    path('update', home, name='home_update'),
    path('upload/', upload_system_info, name='upload_system_info'),  # For uploading data (POST)
]