from django.urls import path
from . import views
from django.contrib.auth import views as auth_views


urlpatterns = [

    path('', views.accueil, name='accueil'),  
    path('classify/', views.classify_image, name='classify_image'),
    

    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('profile/', views.profile, name='profile'),

    path('password_reset/', auth_views.PasswordResetView.as_view(template_name='user/password_reset.html'), name='password_reset'),
    path('password_reset_done/', auth_views.PasswordResetDoneView.as_view(template_name='user/password_reset_done.html'), name='password_reset_done'),
    path('password_reset_confirm/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='user/password_reset_confirm.html'), name='password_reset_confirm'),
    path('password_reset_complete/', auth_views.PasswordResetCompleteView.as_view(template_name='user/password_reset_complete.html'), name='password_reset_complete')

]
