from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import logout,authenticate, login
from django.contrib.auth.decorators import login_required
from .models import *
from .helper_functions import *
from django.contrib import messages
import random


def index(request):
    return render(request, 'user/index.html')


def signup_view(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        
        DOB = request.POST.get("DOB")
        mobile = request.POST.get("mobile")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")
        username=""
        for i in email:
            if i=="@":
                break
            username+=i
            
        password_error = check_strong_password(password)

        if password_error:
            messages.error(request, password_error)
            return redirect("sign_up")

        if password != confirm_password:
            messages.error(request, "Password and confirm password do not match.")
            return redirect("sign_up")

        if Custom_user.objects.filter(email=email).exists():
            messages.error(request, "Email already exists.")
            return redirect("sign_up")

        if Custom_user.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect("sign_up")

        user = Custom_user(
            name=name,
            email=email,
            username=username,
            DOB=DOB,
            mobile=mobile
        )

        user.set_password(password)
        user.save()

        messages.success(request, "Signup successful. Please login.")
        return redirect("login")

    return render(request, "user/signup.html")



def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        print(email,password)
        user = authenticate(request, email=email, password=password)
        print(user)
        if user is not None:
            login(request, user)
            session_id=generating_session_id()
            Chat_Title.objects.create(
                user=user,chat_id=session_id,chat_title="first chat")
            
            messages.success(request, "Login successful.")
            return redirect("chatbot")
        else:
            print("login failed for email:", email)
            messages.error(request, "Invalid email or password.")
            return redirect("login")

    return render(request, "user/login.html")



def logout_view(request):
    logout(request)
    return redirect("login")



@login_required(login_url="login")
def chatbot(request):

    current_chat = Chat_Title.objects.filter(
        user=request.user
    ).last()

    if request.method == "POST":

        user_message = request.POST.get("message")

        if user_message:

            bot_reply = "You said: " + user_message

            Conversation.objects.create(
                chat_id=current_chat,
                user_message=user_message,
                bot_message=bot_reply
            )

        return redirect("chatbot")

    conversation = Conversation.objects.filter(
        chat_id=current_chat
    )

    all_chats = Chat_Title.objects.filter(
        user=request.user
    )

    return render(
        request,
        "user/chatbot.html",
        {
            "conversation": conversation,
            "all_chats": all_chats,
            "current_chat": current_chat
        }
    )



def exit_chat(request):
    session_id = generating_session_id()

    Chat_Title.objects.create(
        user=request.user,
        chat_id=str(session_id),
        chat_title="New Chat"
    )

    return redirect("chatbot")



def conversation(request):
    con=Conversation.objects.all()
    print(con)
    return render(request,"user/conversation.html",{"con":con})


def history_based_on_session(request, session):
    chat = Chat_Title.objects.get(user=request.user,chat_id=session)
    if request.method == "POST":
        user_message = request.POST.get("message")
        if user_message:
            bot_reply = "You said: " + user_message
            Conversation.objects.create(chat_id=chat,user_message=user_message,bot_message=bot_reply)
        return redirect("history_based_on_session",session=session)
    conversation = Conversation.objects.filter(chat_id=chat)
    all_chats = Chat_Title.objects.filter(user=request.user)
    return render(request,"user/chatbot.html",{"conversation": conversation,"all_chats": all_chats,"current_chat": chat})


def chat_title_maker(request,session_id,title):
    # title=None

    Chat_Title.objects.filter(user=request.user,chat_id=session_id).update(chat_title=title)



def chat_title_checker(request):
    a=Chat_Title.objects.all()
    return render(request,"user/chat_title.html",{"data":a})