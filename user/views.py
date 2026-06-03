from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import logout,authenticate, login
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
            session_id=[]
            inter_Session_id=random.randint(10000000,99999999)
            inter_Session_id=str(inter_Session_id)
            for i in inter_Session_id:
                if len(session_id)%3==0:
                    session_id.append("-")
                session_id.append(i)    
            session_id="".join(session_id)
            print(session_id)
            Chat_Title.objects.create(
                user=user,chat_id=session_id,chat_title="Not avaibable")
            
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



def chatbot(request):
    if not request.session.session_key:
        request.session.create()

    session_key = request.session.session_key

    chat_session, created = ChatSession.objects.get_or_create(
        session_key=session_key
    )

    if created or not chat_session.conversation:
        chat_session.conversation = [
            {
                "sender": "bot",
                "message": "Hello! Welcome to the chatbot. How can I help you?"
            }
        ]
        chat_session.save()

    if request.method == "POST":
        user_message = request.POST.get("message")

        if user_message:
            conversation = chat_session.conversation

            conversation.append({
                "sender": "user",
                "message": user_message
            })

            bot_reply = "You said: " + user_message

            conversation.append({
                "sender": "bot",
                "message": bot_reply
            })

            chat_session.conversation = conversation
            chat_session.save()

        return redirect("chatbot")

    return render(request, "user/chatbot.html", {
        "conversation": chat_session.conversation
    })


def exit_chat(request):
    if request.session.session_key:
        ChatSession.objects.filter(
            session_key=request.session.session_key
        ).delete()

    request.session.flush()

    return redirect("chatbot")