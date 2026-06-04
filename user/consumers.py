from channels.generic.websocket import WebsocketConsumer
from .models import Conversation, Chat_Title
import json
from .nlp_bridge import get_response
from .views import chat_title_maker
from .helper_functions import remove_stopwords

class ChatConsumer(WebsocketConsumer):
    def connect(self):
        # self.count=0
        print("connected")
        self.accept()

    def disconnect(self, code=None):
        pass

    def receive(self, text_data=None, bytes_data=None):

        print("Received:", text_data)

        data = json.loads(text_data)

        message = data.get("message")
        chat_id = data.get("chat_id")
        title=remove_stopwords(message)
        # title=None
        # updating chat title according to first message
        chat = Chat_Title.objects.get(chat_id=chat_id)
        if chat.chat_title == "New Chat":
            chat.chat_title = remove_stopwords(message)
            chat.save()
        
        

        print("MESSAGE:", message)
        print("CHAT ID:", chat_id)

        bot_response = get_response(message)

        try:

            chat = Chat_Title.objects.get(chat_id=chat_id)

            Conversation.objects.create(
                chat_id=chat,
                user_message=message,
                bot_message=bot_response
            )

            self.send(text_data=json.dumps({
                "status": "success",
                "bot_response": bot_response
            }))

        except Exception as e:

            print("ERROR:", e)

            self.send(text_data=json.dumps({
                "status": "error",
                "message": str(e)
            }))

    
   