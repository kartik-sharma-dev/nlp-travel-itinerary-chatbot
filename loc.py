import os
import re
import pandas as pd
import spacy
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math


nlp = spacy.load('en_core_web_md')

LABELS = ['A', 'B', 'C', 'D', 'E']

intents = {
    "hotel-keywords": {
        "keywords": ['hotel', 'room', 'stay', 'accommodation', 'lodge', 'resort', 'book']     

    },
    
    "location-keywords": {
        "keywords": ['nearby', 'near', 'close to', 'around', 'some nearby', 'where i can stay', 'i can stay', 'can stay']
    },
    "distance-query-keywords": {
        "keywords": ['distance', 'how far', 'km', 'miles', 'haversine', 'next closest', 'closest', 'nearby']
    },
    "restaurant-keywords": {
        "keywords": ['restaurant', 'food', 'dining', 'eatery', 'cafe', 'diner', 'bistro', 'eat']    
    },
    "taxicab-keywords": {
        "keywords": ['taxi', 'cab', 'ride', 'transport', 'uber', 'lyft', 'taxicab', 'chauffeur']
    }

}



number_words = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
    'nine': '9', 'ten': '10'
}

def detect_intent(query):
    query = query.lower()
    
    distance_specific = ['distance', 'how far', 'km', 'miles', 'haversine']
    if any(kw in query for kw in distance_specific):
        return 'distance'
    
    intent_return = {intent.split('-')[0]: 0 for intent in intents}
    for intent, details in intents.items():
        for keyword in details['keywords']:
            if keyword in query:
                intent_return[intent.split('-')[0]] = True
                break
    return max(intent_return, key=intent_return.get)

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


data, vectorizer, tfidf_matrix = loaddata()

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

    
    person_count = extract_person_count(user_query)
    print(f"Detected person count: {person_count}")

    
    location_query = extract_location(user_query)
    if location_query == '':
        location_query = input("  Please enter a location to find hotels nearby: ").strip()
        if not location_query:
            print("  No location provided. Exiting.")
            return
    

    
    processed_query = preprocess(location_query)
    query_vector    = vectorizer.transform([processed_query])

    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    hotel_data = data[data['hotel-name'].str.strip() != ''].copy()
    hotel_data = hotel_data[hotel_data['hotel-name'] != '0']

    hotel_data['tfidf_score'] = similarities[hotel_data.index]
    hotel_data['fuzz_score']  = hotel_data['processed_place_name'].apply(
        lambda name: calculate_similarity(processed_query, name)
    )
    hotel_data['similarity'] = hotel_data['tfidf_score'] * 0.3 + hotel_data['fuzz_score'] * 0.7

    filtered = hotel_data
    if person_count:
        filtered = hotel_data[
            pd.to_numeric(hotel_data['person aloud per room'], errors='coerce') >= person_count
        ]

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
def extract_entities_bio(query):
    
    doc = nlp(query)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ('GPE', 'LOC', 'FAC'):
            entities.append(ent.text.lower().strip())
    return entities

def handle_distance_query(query, data):
    
    print(f"\nUser Query: {query}")

    entities = extract_entities_bio(query)
    print(f"Detected entities: {entities}")

    if len(entities) < 2:
        print("  spaCy found fewer than 2 entities, trying fallback...")
        fallback = extract_location(query)
        parts = [p.strip() for p in fallback.split(' and ') if p.strip()]
        entities = parts
        print(f"  Fallback entities: {entities}")

    if len(entities) < 2:
        print("  Could not detect two locations in your query.")
        print("  Try: 'distance between <place1> and <place2>'")
        return

    processed_a = preprocess(entities[0])
    processed_b = preprocess(entities[1])

    ranked_a = score_and_rank(processed_a, data)
    ranked_b = score_and_rank(processed_b, data)

    place_a = ranked_a.iloc[0]
    place_b = ranked_b.iloc[0]

    print(f"  Matched A: {place_a['PlaceName']} (score: {ranked_a.iloc[0]['final_score']:.1f})")
    print(f"  Matched B: {place_b['PlaceName']} (score: {ranked_b.iloc[0]['final_score']:.1f})")

    dist = haversine(
        place_a['Latitude'], place_a['Longitude'],
        place_b['Latitude'], place_b['Longitude']
    )

    print()
    print("=" * 52)
    print(f"  Place A  : {place_a['PlaceName']}")
    print(f"             {place_a['State/Province']}, {place_a['Country']}")
    print(f"             Lat {place_a['Latitude']}  |  Lon {place_a['Longitude']}")
    print()
    print(f"  Place B  : {place_b['PlaceName']}")
    print(f"             {place_b['State/Province']}, {place_b['Country']}")
    print(f"             Lat {place_b['Latitude']}  |  Lon {place_b['Longitude']}")
    print()
    print(f"  Distance : {dist} km")
    print("=" * 52)

    return {
        'place_a': place_a,
        'place_b': place_b,
        'distance_km': dist
    }


if __name__ == '__main__':
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
        if not query:
            continue

        intent = detect_intent(query)
        print(f"Detected intent: {intent}")

        if intent == 'hotel':
            handle_query(query, data, vectorizer, tfidf_matrix)
        elif intent == 'distance':
            handle_distance_query(query, data)
        else:
            location(query, data)


