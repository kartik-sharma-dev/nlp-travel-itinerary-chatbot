import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess
from query_utils import extract_location, extract_entities_bio, extract_days, extract_budget, extract_itinerary_location
from scoring import score_and_rank, build_chain, haversine, LABELS, calculate_similarity
import random
import re
from preprocess import nlp
from detect_intent import session


def fmt_coord(val):
    return f"{val:.4f}" if not (isinstance(val, float) and math.isnan(val)) else "N/A"


def resolve_location(query):
  
    location_query = extract_location(query)

    if not location_query and session["last_location"]:
        print(f"  (Using last known location: {session['last_location']})")
        location_query = session["last_location"]

    return location_query


def handle_greeting(query):
    greet_responses = [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Hey! Hope you're having a great day.",
        "Greetings! How may I assist you?",
        "Welcome! What would you like help with?",
        "Hi! Feel free to ask me anything.",
        "Hey there! How can I assist you today?",
        "Good to see you! What can I help you with?",
    ]
    doc = nlp(query.strip())

    if any(token.pos_ == "INTJ" for token in doc):
        print(random.choice(greet_responses))
        return

    if len(doc) <= 5:
        root_tokens = [t for t in doc if t.dep_ == "ROOT"]
        if root_tokens:
            root = root_tokens[0]
            subjects = [t for t in root.children if t.dep_ in ("nsubj", "nsubj:pass")]
            if any(t.lemma_ in ("you", "it", "thing") for t in subjects):
                print(random.choice(greet_responses))
                return

    print(random.choice(greet_responses))


def handle_hotel_query(query, data, hotel_df, vectorizer, tfidf_matrix):
    print(f"\nUser Query: {query}")

    location_query = resolve_location(query)
    if not location_query:
        location_query = input("  Please enter a location to find hotels nearby: ").strip()
        if not location_query:
            print("  No location provided. Exiting.")
            return

    # save to session
    session["last_location"] = location_query

    processed_query = preprocess(location_query)
    query_vector    = vectorizer.transform([processed_query])

    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    hotel_df = hotel_df.copy()
    hotel_df["tfidf_score"] = similarities[hotel_df.index]
    hotel_df["fuzz_score"]  = hotel_df["proc_location"].apply(
        lambda name: calculate_similarity(processed_query, name)
    )
    hotel_df["similarity"]  = hotel_df["tfidf_score"] * 0.3 + hotel_df["fuzz_score"] * 0.7

    results = hotel_df.sort_values("similarity", ascending=False).head(5)

    if results.empty:
        print(f"\n  No hotels found near '{location_query}'.")
        return None

    print("\n" + "=" * 52)
    for i, (_, row) in enumerate(results.iterrows()):
        print(f"\n  Hotel {i+1}  : {row['landmark']}")
        print(f"  Place     : {row['location']}")
        print(f"  State     : {row['state']}, {row['country']}")
        print(f"  Rating    : {row['ratings']}")
        print(f"  Reviews   : {row['total_reviews']}")
        print(f"  Coords    : {fmt_coord(row['lat_location'])}, {fmt_coord(row['lon_location'])}")
    print("=" * 52)

    # save results to session
    session["last_results"] = results

    return results


def handle_restaurant_query(query, data, restaurant_df, vectorizer, tfidf_matrix):
    print(f"\nUser Query: {query}")

    location_query = resolve_location(query)
    if not location_query:
        location_query = input("  Please enter a location to find restaurants nearby: ").strip()
        if not location_query:
            print("  No location provided. Exiting.")
            return

    session["last_location"] = location_query

    processed_query = preprocess(location_query)
    query_vector    = vectorizer.transform([processed_query])

    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    restaurant_df = restaurant_df.copy()
    restaurant_df["tfidf_score"] = similarities[restaurant_df.index]
    restaurant_df["fuzz_score"]  = restaurant_df["proc_location"].apply(
        lambda name: calculate_similarity(processed_query, name)
    )
    restaurant_df["similarity"]  = restaurant_df["tfidf_score"] * 0.3 + restaurant_df["fuzz_score"] * 0.7

    results = restaurant_df.sort_values("similarity", ascending=False).head(5)

    if results.empty:       
        print(f"\n  No restaurants found near '{location_query}'.")
        return None

    print("\n" + "=" * 52)
    for i, (_, row) in enumerate(results.iterrows()):
        print(f"\n  Restaurant {i+1} : {row['landmark']}")
        print(f"  Place          : {row['location']}")
        print(f"  State          : {row['state']}, {row['country']}")
        print(f"  Rating         : {row['ratings']}")
        print(f"  Reviews        : {row['total_reviews']}")
        print(f"  Coords         : {fmt_coord(row['lat_location'])}, {fmt_coord(row['lon_location'])}")
    print("=" * 52)

    session["last_results"] = results

    return results


def clean_entity(text):
    noise = r'\b(the|a|an|distance|between|from|to|tell|me|hey|find|what|is)\b'
    cleaned = re.sub(noise, '', text, flags=re.IGNORECASE).strip()
    return cleaned


def handle_distance_query(query, data, vectorizer, tfidf_matrix):
    print(f"\nUser Query: {query}")

    entities = extract_entities_bio(query)

    if len(entities) < 2:
        fallback = extract_location(query)
        parts    = [p.strip() for p in fallback.split(' and ') if p.strip()]
        entities = parts

    if len(entities) < 2:
        print("  Could not detect two locations in your query.")
        print("  Try: 'distance between <place1> and <place2>'")
        return

    entity_a = clean_entity(entities[0])
    entity_b = clean_entity(entities[1])

    ranked_a = score_and_rank(preprocess(entity_a), data, vectorizer, tfidf_matrix)
    ranked_b = score_and_rank(preprocess(entity_b), data, vectorizer, tfidf_matrix)

    place_a = ranked_a.iloc[0]
    place_b = ranked_b.iloc[0]

    dist = haversine(
        place_a["lat_location"], place_a["lon_location"],
        place_b["lat_location"], place_b["lon_location"]
    )

    print()
    print("=" * 52)
    print(f"  Place A  : {place_a['location']} — {place_a['landmark']}")
    print(f"             {place_a['state']}, {place_a['country']}")
    print(f"             Lat {fmt_coord(place_a['lat_location'])}  |  Lon {fmt_coord(place_a['lon_location'])}")
    print()
    print(f"  Place B  : {place_b['location']} — {place_b['landmark']}")
    print(f"             {place_b['state']}, {place_b['country']}")
    print(f"             Lat {fmt_coord(place_b['lat_location'])}  |  Lon {fmt_coord(place_b['lon_location'])}")
    print()
    print(f"  Distance : {dist} km")
    print("=" * 52)

    return {"place_a": place_a, "place_b": place_b, "distance_km": dist}


def find_nearest_in_df(lat, lon, df):
    if df.empty:
        return None, None
    valid = df[df['lat_location'].notna() & df['lon_location'].notna()].copy()
    if valid.empty:
        return None, None
    valid['_dist'] = valid.apply(
        lambda r: haversine(lat, lon, r['lat_location'], r['lon_location']), axis=1
    )
    valid = valid[valid['_dist'] > 0]
    if valid.empty:
        return None, None
    idx = valid['_dist'].idxmin()
    return valid.loc[idx], valid.loc[idx, '_dist']


def handle_itinerary_query(query, data, hotel_df, restaurant_df, vectorizer, tfidf_matrix):
    print(f"\nUser Query: {query}")

   
    location_query = extract_itinerary_location(query)

    if not location_query:
        location_query = input("  Please enter a destination for your trip: ").strip()
        if not location_query:
            print("  No destination provided.")
            return

    days = extract_days(query) or session.get("trip_days") or 2
    budget = extract_budget(query)
    if budget is None:
        budget = session["preferences"]["budget"]

    session["last_location"] = location_query
    session["trip_days"] = days
    if budget is not None:
        session["preferences"]["budget"] = budget

    processed = preprocess(location_query)
    ranked = score_and_rank(processed, data, vectorizer, tfidf_matrix)
    best = ranked.iloc[0]

    if best["final_score"] < 15:
        print(f"\n  Could not find '{location_query}'. Try a city or place name.")
        return

    # Find best matching place (type=place) in the same state to start the chain
    state_places = ranked[(ranked['state'] == best['state']) & (ranked['type'] == 'place')]
    if state_places.empty:
        print(f"\n  No places to visit found in '{location_query}'.")
        return
    best_place = state_places.iloc[0]

    stops_per_day = 2
    chain = build_chain(best_place, data, max_stops=days * stops_per_day, min_distance=5, place_only=True)
    day_groups = [chain[i:i + stops_per_day] for i in range(0, len(chain), stops_per_day)][:days]

    print()
    print("=" * 52)
    print(f"  Trip Plan    : {days}-Day Itinerary")
    # Fix 2: show only the clean extracted location name, not the full raw query
    print(f"  Destination  : {location_query.title()}")
    print(f"  State        : {best_place['state']}, {best_place['country']}")
    if budget is not None:
        daily = budget / days
        print(f"  Budget       : ₹{budget:,.0f}  (~₹{daily:,.0f}/day)")

    # Track booked hotel/restaurant names so they're not double-booked across days
    visited_hotels = set()  # hotel location names already allocated
    visited_rests  = set()  # restaurant location names already allocated
    visited_places = set()  # landmark names of visited sights (for display dedup only)

    for day_num, stops in enumerate(day_groups, 1):
        if not stops:
            continue
        print(f"\n  --- Day {day_num} ---")
        for stop in stops:
            visited_places.add(stop['landmark'])
            print(f"    * Visit : {stop['landmark']}")
            print(f"              {stop['location']}, {stop['state']}")
            print(f"              Coords: {fmt_coord(stop['lat_location'])}, {fmt_coord(stop['lon_location'])}")

        ref_lat       = stops[0]['lat_location']
        ref_lon       = stops[0]['lon_location']
        stop_location = stops[0]['location']
        stop_state    = stops[0]['state']

        MAX_STAY_KM = 30  # don't show hotels/restaurants further than this

        # prefer hotels near the same landmark as the visited stop; fall back to same state
        # hotel rows store the nearby place name in 'landmark', not 'location'
        available_hotels = hotel_df[~hotel_df['location'].isin(visited_hotels)]
        city_hotels      = available_hotels[available_hotels['landmark'] == stop_location]
        hotel_row, h_dist = find_nearest_in_df(ref_lat, ref_lon, city_hotels)
        if hotel_row is None or h_dist > MAX_STAY_KM:
            state_hotels = available_hotels[available_hotels['state'] == stop_state]
            hotel_row, h_dist = find_nearest_in_df(ref_lat, ref_lon, state_hotels)
        if hotel_row is not None and h_dist <= MAX_STAY_KM:
            visited_hotels.add(hotel_row['location'])
            print(f"\n    Stay  : {hotel_row['location']}")
            print(f"              near {hotel_row['landmark']}")
            print(f"              Rating {hotel_row['ratings']}  |  {hotel_row['total_reviews']} reviews")
            print(f"              {h_dist:.1f} km from Day {day_num} base")
        else:
            print(f"\n    Stay  : No hotel found within {MAX_STAY_KM} km")

        # prefer restaurants near the same landmark as the visited stop; fall back to same state
        available_rests = restaurant_df[~restaurant_df['location'].isin(visited_rests)]
        city_rests      = available_rests[available_rests['landmark'] == stop_location]
        rest_row, r_dist = find_nearest_in_df(ref_lat, ref_lon, city_rests)
        if rest_row is None or r_dist > MAX_STAY_KM:
            state_rests = available_rests[available_rests['state'] == stop_state]
            rest_row, r_dist = find_nearest_in_df(ref_lat, ref_lon, state_rests)
        if rest_row is not None and r_dist <= MAX_STAY_KM:
            visited_rests.add(rest_row['location'])
            print(f"\n    Eat   : {rest_row['location']}")
            print(f"              near {rest_row['landmark']}")
            print(f"              Rating {rest_row['ratings']}  |  {rest_row['total_reviews']} reviews")
            print(f"              {r_dist:.1f} km from Day {day_num} base")
        else:
            print(f"\n    Eat   : No restaurant found within {MAX_STAY_KM} km")

    print()
    print("=" * 52)
    session["last_results"] = chain
    return chain


def handle_location_query(query, data, vectorizer, tfidf_matrix):
    if not query:
        return None

    processed = preprocess(query)
    ranked    = score_and_rank(processed, data, vectorizer, tfidf_matrix)
    best      = ranked.iloc[0]

    if best["final_score"] < 15:
        print(f"\n  Could not find a location matching '{query}'. Try a city or place name.")
        return None

    # save to session
    session["last_location"] = query

    chain = build_chain(best, data)

    print()
    print("=" * 52)
    for i, row in enumerate(chain):
        label = LABELS[i]
        print(f"  Location {label}: {row['location']}")
        print(f"             {row['state']}, {row['country']}")
        print(f"             Lat {fmt_coord(row['lat_location'])}  |  Lon {fmt_coord(row['lon_location'])}")
        if i > 0:
            prev = chain[i - 1]
            dist = haversine(
                prev["lat_location"], prev["lon_location"],
                row["lat_location"],  row["lon_location"]
            )
            print(f"             Distance {LABELS[i-1]} → {label}: {dist} km")
        if i < len(chain) - 1:
            print("             ↓")
    print("=" * 52)

    # save results to session
    session["last_results"] = chain

    return chain