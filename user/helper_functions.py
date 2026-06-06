import re
import random
import spacy
from .models import *
from transformers import pipeline

def check_strong_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters long."

    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."

    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."

    if not re.search(r"\d", password):
        return "Password must contain at least one number."

    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return "Password must contain at least one special character."

    return None


def generating_session_id():
    session_id=[]
    inter_Session_id=random.randint(10000000,99999999)
    inter_Session_id=str(inter_Session_id)
    for i in inter_Session_id:
        if len(session_id)%3==0:
            session_id.append("-")
        session_id.append(i)    
    session_id="".join(session_id)
    print(session_id)
    return session_id



# Load once when application starts
nlp = spacy.load("en_core_web_md")

def remove_stopwords(text):
    """
    Removes stop words and punctuation from text.
    """

    doc = nlp(text)

    filtered_tokens = [
        token.text
        for token in doc
        if not token.is_stop and not token.is_punct
    ]

    return " ".join(filtered_tokens)



def chat_title_edit(request,old_title,new_title):
    # title=None

    Chat_Title.objects.filter(user=request.user,title=old_title).update(chat_title=new_title)


def delete_session(session_id):
    Chat_Title.objects.filter(chat_id=session_id).delete()    





def _clean_for_summary(text):
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[=*|\\-]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def summary_function(session_id):
    obj = Chat_Title.objects.get(chat_id=session_id)
    chat = Conversation.objects.filter(chat_id=obj)

    text = _clean_for_summary(" ".join(
        f"User asked: {msg.user_message}. Bot replied: {msg.bot_message}"
        for msg in chat
    ))

    if len(text.split()) < 10:
        return "Not enough conversation history to summarize."

    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(text, max_length=100, min_length=10, do_sample=False)
    return summary[0]["summary_text"]
    