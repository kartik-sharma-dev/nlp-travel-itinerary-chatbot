from django.urls import path
from user import views
from django.contrib import admin

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.index, name="index"),
    path("sign-up/", views.signup_view, name="sign_up"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("exit/", views.exit_chat, name="exit_chat"),
    path("chat/", views.chatbot, name="chatbot"),
    path("conversation/", views.conversation, name="conversation"),
    path("chat/<str:session>/", views.history_based_on_session, name="history_based_on_session"),
    path("chat_title/",views.chat_title_checker,name="chat"),
]