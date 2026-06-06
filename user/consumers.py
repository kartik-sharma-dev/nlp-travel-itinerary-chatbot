from channels.generic.websocket import WebsocketConsumer
from .models import Conversation, Chat_Title
import json
from .nlp_bridge import get_response, _new_session
from .views import chat_title_maker
from .helper_functions import remove_stopwords,summary_function



class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.bot_session = _new_session()
        print("connected")
        self.accept()

    def disconnect(self, code=None):
        pass

    def receive(self, text_data=None, bytes_data=None):

        print("Received:", text_data)

        data = json.loads(text_data)
        action = data.get("action")
        chat_id = data.get("chat_id")

        if action == "update_title":
            new_title = data.get("new_title", "").strip()
            if new_title and chat_id:
                Chat_Title.objects.filter(chat_id=chat_id).update(chat_title=new_title)
                self.send(text_data=json.dumps({"status": "title_updated"}))
            return

        message = data.get("message")

        print("MESSAGE:", message)
        print(type(message))
        if "summary" in message or "last" in message:
            self.send(text_data=json.dumps({
                "status": "success",
                "bot_response": summary_function(chat_id)#how???????????
            }))
            

        print("CHAT ID:", chat_id)

        bot_response = get_response(message, self.bot_session)

        try:

            chat = Chat_Title.objects.get(chat_id=chat_id)

            # update chat title from first message
            if chat.chat_title in ("New Chat", "first chat"):
                chat.chat_title = remove_stopwords(message)
                chat.save()

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

    
   