import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess
from query_utils import extract_person_count, extract_location, extract_entities_bio
from scoring import score_and_rank, build_chain, haversine, LABELS, calculate_similarity
import random
from preprocess import nlp


def _fmt_coord(val):
    return f"{val:.4f}" if not (isinstance(val, float) and math.isnan(val)) else "N/A"


def handle_greeting(query):
    greet_responses = [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Hey! Hope you're having a great day.",
        "Greetings! How may I assist you?",
        "Welcome! What would you like help with?",
        "Hi! Feel free to ask me anything.",
        "Hey there! How can I assist you today?",
        "Good to see you! What can I help you with?"]
    doc = nlp(query.strip())

    if any(token.pos_ == "INTJ" for token in doc):
        print(random.choice(greet_responses))
        return

    # Short inputs with greeting-like verb structure: "how are you", "what's up"
    if len(doc) <= 5:
        root_tokens = [t for t in doc if t.dep_ == "ROOT"]
        if root_tokens:
            root = root_tokens[0]
            subjects = [t for t in root.children if t.dep_ in ("nsubj", "nsubj:pass")]
            if any(t.lemma_ in ("you", "it", "thing") for t in subjects):
                print(random.choice(greet_responses))
                return

    print(random.choice(greet_responses))


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
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    hotel_data = data[data['hotel-name'].str.strip() != ''].copy()
    hotel_data = hotel_data[hotel_data['hotel-name'] != '0']
    has_details = (
        (hotel_data['rating'].astype(str).str.strip() != '') |
        (hotel_data['price'].astype(str).str.strip() != '') |
        (hotel_data['room type'].astype(str).str.strip() != '')
    )
    hotel_data = hotel_data[has_details]

    hotel_data['tfidf_score'] = similarities[hotel_data.index]
    hotel_data['fuzz_score'] = hotel_data['processed_place_name'].apply(
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


def handle_distance_query(query, data, vectorizer, tfidf_matrix):
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

    ranked_a = score_and_rank(processed_a, data, vectorizer, tfidf_matrix)
    ranked_b = score_and_rank(processed_b, data, vectorizer, tfidf_matrix)

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
    print(f"             Lat {_fmt_coord(place_a['Latitude'])}  |  Lon {_fmt_coord(place_a['Longitude'])}")
    print()
    print(f"  Place B  : {place_b['PlaceName']}")
    print(f"             {place_b['State/Province']}, {place_b['Country']}")
    print(f"             Lat {_fmt_coord(place_b['Latitude'])}  |  Lon {_fmt_coord(place_b['Longitude'])}")
    print()
    print(f"  Distance : {dist} km")
    print("=" * 52)

    return {'place_a': place_a, 'place_b': place_b, 'distance_km': dist}


def location(query, data, vectorizer, tfidf_matrix):
    if not query:
        return None
    processed = preprocess(query)
    ranked = score_and_rank(processed, data, vectorizer, tfidf_matrix)
    best = ranked.iloc[0]

    if best['final_score'] < 15:
        print(f"\n  Could not find a location matching '{query}'. Try a city or place name.")
        return None

    chain = build_chain(best, data)

    print()
    print("=" * 52)
    for i, row in enumerate(chain):
        label = LABELS[i]
        print(f"  Location {label}: {row['PlaceName']}")
        print(f"             {row['State/Province']}, {row['Country']}")
        print(f"             Lat {_fmt_coord(row['Latitude'])}  |  Lon {_fmt_coord(row['Longitude'])}")
        if i > 0:
            prev = chain[i - 1]
            dist = haversine(prev['Latitude'], prev['Longitude'], row['Latitude'], row['Longitude'])
            print(f"             Distance {LABELS[i-1]} → {label}: {dist} km")
        if i < len(chain) - 1:
            print(f"             ↓")
    print("=" * 52)
