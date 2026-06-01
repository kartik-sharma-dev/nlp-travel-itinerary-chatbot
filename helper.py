import os
import re
import numpy as np
import pandas as pd
import spacy
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from loc import detect_intent,intents,preprocess,loaddata

nlp = spacy.load('en_core_web_sm')

intents = {
    "hotel-keywords": {
        "keywords": ['hotel', 'room', 'stay', 'accommodation', 'lodge', 'resort', 'book']     

    },
    
    "location-keywords": {
        "keywords": ['nearby', 'near', 'close to', 'around', 'some nearby', 'where i can stay', 'i can stay', 'can stay']
    },
    "restaurant-keywords": {
        "keywords": ['restaurant', 'food', 'dining', 'eatery', 'cafe', 'diner', 'bistro', 'eat']    
    },
    "distance-query-keywords": {
        "keywords": ['distance', 'how far', 'km', 'miles', 'haversine', 'next closest', 'closest', 'nearby']
    },
    "taxicab-keywords": {
        "keywords": ['taxi', 'cab', 'ride', 'transport', 'uber', 'lyft', 'taxicab', 'chauffeur']
    },
    "intent-return":{
        'nearby': False,'restaurant': False,'hotel': False,'taxi': False
    }
    
}



number_words = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
    'nine': '9', 'ten': '10'
}

def preprocess(text):
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

def detect_intent(user_input):
    query = preprocess(user_input).lower()
    intent_return = {intent.split('-')[0]: 0 for intent in intents}
    for intent, details in intents.items():
        for keyword in details['keywords']:
            if keyword in query:
                intent_return[intent.split('-')[0]] = True
                break
    return max(intent_return, key=intent_return.get)


def loaddata():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'location.csv')
    filepath2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hotel_places_directory.csv')
    data1 = pd.read_csv(filepath)
    data2 = pd.read_csv(filepath2)
    data2 = data2.rename(columns={
        'country': 'Country',
        'state': 'State/Province',
        'place name': 'PlaceName',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
    })
    data = pd.concat([data1, data2])
    data.drop_duplicates(inplace=True)
    data = data.fillna('')
    data['processed_country'] = data['Country'].apply(preprocess)
    data['processed_state'] = data['State/Province'].apply(preprocess)
    data['processed_place_name'] = data['PlaceName'].apply(preprocess)
    data['combined'] = data['processed_place_name'] + " " + data['processed_state'] + " " + data['processed_country']
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    tfidf_matrix = vectorizer.fit_transform(data['processed_place_name'])
    return data, vectorizer, tfidf_matrix


def calculate_similarity(query, target):
    if not query or not target:
        return 0
    query_tokens = query.split()
    target_tokens = target.split()
    scores = []
    for t in target_tokens:
        bestmatch = max([fuzz.ratio(t, q) for q in query_tokens]) if query_tokens else 0
        scores.append(bestmatch)
    return sum(scores) / len(scores) if scores else 0

data, vectorizer, tfidf_matrix = loaddata()

def score_and_rank(processed, data):
    def fuzzyscore(row):
        place = calculate_similarity(processed, row['processed_place_name']) * 0.70
        state  = calculate_similarity(processed, row['processed_state'])      * 0.20
        country = calculate_similarity(processed, row['processed_country'])   * 0.10
        return place + state + country

    scored = data.copy()
    scored['fuzz_score'] = scored.apply(fuzzyscore, axis=1)
    query_vec = vectorizer.transform([processed])
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten() * 100
    scored['tfidf_score'] = cosine_scores
    scored['final_score'] = scored['fuzz_score'] * 0.7 + scored['tfidf_score'] * 0.3
    return scored.sort_values('final_score', ascending=False)



