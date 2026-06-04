from channels.generic.websocket import WebsocketConsumer
from .models import Conversation, Chat_Title
import json


class ChatConsumer(WebsocketConsumer):

    def connect(self):
        print("connected")
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):

        data = json.loads(text_data)

        message = data.get("message")
        chat_id = data.get("chat_id")

        try:
            chat = Chat_Title.objects.get(id=chat_id)

            conversation = Conversation(
                chat_id=chat,
                user_message=message,
                bot_message=None
            )

            conversation.save()

            self.send(text_data=json.dumps({
                "status": "success",
                "message": message
            }))

        except Chat_Title.DoesNotExist:

            self.send(text_data=json.dumps({
                "status": "error",
                "message": "Chat not found"
            }))