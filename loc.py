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

LABELS = ['A', 'B', 'C', 'D', 'E']

number_words = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
    'nine': '9', 'ten': '10'
}

def detect_intent(query):
    hotel_keywords = ['hotel', 'room', 'stay', 
                      'accommodation', 'lodge', 'resort', 'booking']
    for keyword in hotel_keywords:
        if keyword in query.lower():
            return 'hotel'
    return 'location'

# Preprocesses text by converting to lowercase, replacing number words with digits, removing stop words and punctuation, and lemmatizing.
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


def loaddata():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'location.csv')
    filepath2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hotel_places_directory.csv')
    data1 = pd.read_csv(filepath)
    data2 = pd.read_csv(filepath2)
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


data, vectorizer, tfidf_matrix = loaddata()

# Extracts the number of people from the query by replacing number words with digits and using regex to find numeric patterns followed by relevant keywords.
def extract_person_count(query):
    
    for word, digit in number_words.items():
        query = re.sub(rf'\b{word}\b', digit, query.lower())
    
    match = re.search(r'\b(\d+)\s+(people|persons?|guests?|adults?|members?)\b', query)
    if match:
        return int(match.group(1))
    return None


def extract_location(query):
    fillers = sorted([
        'i am near', 'i am at', 'i am', 'can you tell me', 'can you tell',
        'some nearby hotel', 'some nearby', 'nearby hotel', 'where i can stay',
        'i can stay', 'can stay', 'close to', 'i need', 'a room',
        'around', 'nearby', 'near', 'people', 'persons', 'guests',
        'hotel', 'some', 'where', 'stay', 'tell', 'for',
        'hy', 'hi', 'hey', 'hello', 'me', 'i', 'can'
    ], key=len, reverse=True)

    query = query.lower()
    for filler in fillers:
        query = re.sub(rf'\b{re.escape(filler)}\b', ' ', query)

    query = re.sub(r'\s+', ' ', query).strip()
    return query


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


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2)**2
    return round(R * 2 * math.asin(math.sqrt(a)), 2)


def next_closest(lat, lon, nearby, visited_coords):
    col = 'dist_tmp'
    distances = []
    for _, row in nearby.iterrows():
        coord = (row['Latitude'], row['Longitude'])
        if coord in visited_coords:
            distances.append(None)
        else:
            distances.append(haversine(lat, lon, row['Latitude'], row['Longitude']))
    nearby = nearby.copy()
    nearby[col] = distances
    valid = nearby.dropna(subset=[col])
    valid = valid[valid[col] > 0]
    if valid.empty:
        return None, None
    best_idx = valid[col].idxmin()
    return valid.loc[best_idx], valid.loc[best_idx, col]


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


def build_chain(best_row, data, max_stops=None):
    if max_stops is None:
        max_stops = len(LABELS)
    state_name = best_row['State/Province']
    nearby = data[data['State/Province'] == state_name].copy()
    chain = [best_row]
    visited = {(best_row['Latitude'], best_row['Longitude'])}

    for _ in range(max_stops - 1):
        prev = chain[-1]
        row, _ = next_closest(prev['Latitude'], prev['Longitude'], nearby, visited)
        if row is None:
            break
        chain.append(row)
        visited.add((row['Latitude'], row['Longitude']))
    return chain
def handle_query(user_query, data, vectorizer, tfidf_matrix):

    print(f"\nUser Query: {user_query}")

    # step 1 - extract person count
    person_count = extract_person_count(user_query)
    print(f"Detected person count: {person_count}")

    # step 2 - extract location
    location_query = extract_location(user_query)
    print(f"Detected location: {location_query}")

    # step 3 - preprocess and vectorize the location
    processed_query = preprocess(location_query)
    query_vector    = vectorizer.transform([processed_query])

    # step 4 - cosine similarity against all place names
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # step 5 - filter hotel rows and assign similarity before filtering
    hotel_data = data[data['hotel-name'].str.strip() != ''].copy()
    hotel_data = hotel_data[hotel_data['hotel-name'] != '0']
    hotel_data['similarity'] = similarities[hotel_data.index]

    # step 6 - filter by person count if detected
    filtered = hotel_data
    if person_count:
        filtered = hotel_data[
            pd.to_numeric(hotel_data['person aloud per room'], errors='coerce') >= person_count
        ]

    # step 7 - sort by similarity and return top 5
    results = filtered.sort_values('similarity', ascending=False).head(5)

    print("\n" + "=" * 52)
    for i, (_, row) in enumerate(results.iterrows()):
        print(f"\n  Hotel {i+1}  : {row['hotel-name']}")
        print(f"  Place     : {row['PlaceName']}")
        print(f"  Room Type : {row['room type']}")
        print(f"  Capacity  : {row['person aloud per room']} people")
        print(f"  Rating    : {row['rating']}")
        print(f"  Price     : ${row['price']}")
    print("=" * 52)

    return results

def location(query, data):
    if not query:
        return None
    processed = preprocess(query)
    ranked = score_and_rank(processed, data)
    best = ranked.iloc[0]
    chain = build_chain(best, data)

    print()
    print("=" * 52)
    for i, row in enumerate(chain):
        label = LABELS[i]
        print(f"  Location {label}: {row['PlaceName']}")
        print(f"             {row['State/Province']}, {row['Country']}")
        print(f"             Lat {row['Latitude']}  |  Lon {row['Longitude']}")
        if i > 0:
            prev = chain[i - 1]
            dist = haversine(prev['Latitude'], prev['Longitude'], row['Latitude'], row['Longitude'])
            print(f"             Distance {LABELS[i-1]} → {label}: {dist} km")
        if i < len(chain) - 1:
            print(f"             ↓")
    print("=" * 52)





if __name__ == '__main__':
    query = input("Enter your question: ")
    intent = detect_intent(query)
    if intent == 'hotel':
        handle_query(query, data, vectorizer, tfidf_matrix)
    else:
        location(query, data)


