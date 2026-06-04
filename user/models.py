from django.contrib.auth.models import AbstractUser
from django.db import models

class BaseModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)
    class Meta:
        abstract = True

class Custom_user(AbstractUser):
    name=models.CharField(max_length=255, null=True, blank=True)
    email=models.EmailField(unique=True)
    username=models.CharField(max_length=255, unique=True)
    DOB=models.DateField(null=True, blank=True)
    mobile=models.CharField(max_length=20, null=True, blank=True)
    password=models.CharField(max_length=255)
    
    
class Chat_Title(models.Model):
    user=models.ForeignKey(Custom_user, on_delete=models.CASCADE)
    chat_id=models.CharField(max_length=255, unique=True)    
    chat_title=models.CharField(max_length=50)
    
class Conversation(models.Model):
    chat_id=models.ForeignKey(Chat_Title, on_delete=models.CASCADE)
    user_message=models.TextField()
    bot_message=models.TextField()    

    def __str__(self):
        return f"Chat {self.chat_id.chat_id}: {self.user_message[:20]}..."
    
class ChatSession(models.Model):
    session_key = models.CharField(max_length=255, unique=True)
    conversation = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.session_key