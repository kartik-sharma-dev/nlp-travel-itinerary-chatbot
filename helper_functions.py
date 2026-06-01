import os
import re
import numpy as np
import pandas as pd
import spacy
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

nlp = spacy.load('en_core_web_sm')


def check_greets(query):
    query=query.lower()
    greetings = ["hi", "hello", "hey", "hiya", "yo", "sup","good morning", "good afternoon","good evening", "good night","greetings", "howdy", "namaste","namaskar", "salaam"]
    understanding_quote=["ok","got it","gotit","done","yes"]
    questioning_quote=["hi how are you","hello how are you","hey what's up","good morning how are you","hi there how's it going"]
    for greet in greetings:
        if query==greet or query in greet:
            return f"{query}.I am chabox with limits,so can you be more specific with the things."
    for q in questioning_quote:
        if q==query or query in q:
            return "thanks for asking please  start with your query"    
    for u in understanding_quote:
        if u==query:
            return f"thanks for understanding"   
    return True     

def preprocess(text):
    number_words = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
    'nine': '9', 'ten': '10'}
    if not isinstance(text, str):
        text = str(text)
    for word, num in number_words.items():
        text = text.replace(word, num)
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        if token.like_num:               
            tokens.append(token.text)
        elif not token.is_stop:
            tokens.append(token.text)   
    return " ".join(tokens)
# {"hotel":False ,"nearby":False,"restro":fslse}

def detect_intent(query):
    hotel_keywords = ['hotel', 'room', 'stay', 
                      'accommodation', 'lodge', 'resort', 'booking']
    for keyword in hotel_keywords:
        if keyword in query.lower():
            return 'hotel'
    return 'location'

def extract_named_entities(text):
    """
    Extract named entities from text.

    Args:
        text (str): Input text

    Returns:
        list: List of tuples (entity_text, entity_label)
    """
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    return entities