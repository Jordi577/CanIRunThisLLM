from django.urls import path
from .views import home, upload_system_info, stop_chart_view, update_table_view

urlpatterns = [
    path('', home, name='home'),
    path('update/', home, name='home_update'),
    path('upload/', upload_system_info, name='upload_system_info'), 
    path('stop-chart/', stop_chart_view, name='stop_chart'),
    path('update_table/', update_table_view, name='update_table'),
]