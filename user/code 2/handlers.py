import csv
import math
import os
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess
from query_utils import extract_location, extract_entities_bio, extract_days, extract_budget, extract_itinerary_location
from scoring import score_and_rank, build_chain, haversine, LABELS, calculate_similarity
import random
import re
import string

_GREET_RESPONSES = [
    "Hello! How can I help you today?",
    "Hi there! What can I do for you?",
    "Hey! Hope you're having a great day.",
    "Greetings! How may I assist you?",
    "Welcome! What would you like help with?",
    "Hi! Feel free to ask me anything.",
    "Hey there! How can I assist you today?",
    "Good to see you! What can I help you with?",
]

_NOISE_WORDS = {
    "nearby", "here", "around", "hotel", "hotels", "good", "some",
    "restaurant", "restaurants", "restaraunt", "restaraunts", "find",
    "me", "a", "an", "the", "in", "at", "to", "for", "any",
    "pure", "veg", "vegetarian", "nonveg", "food", "eat", "best",
    "cheap", "luxury", "top", "rated", "spend", "spending",
    "night", "nights", "stay", "budget", "rupees", "rs", "inr", "₹",
    "places", "place", "dining", "cafe", "diner", "bistro"
}

_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "real_landmark_locations.csv")

# ============================================================================
# UTILITY FUNCTIONS FOR FORMATTING
# ============================================================================

def save_geocoded_to_csv(name, lat, lon):
    """Append a geocoded place to the CSV so the next session indexes it directly."""
    try:
        with open(_CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([name.title(), name.title(), "NA", "NA", lon, lat, "place", "NA", "NA"])
    except Exception as e:
        pass  # Silently fail


def get_lat_lon(location):
    """Fetch lat/lon from Nominatim for any place name not in the dataset."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location, "format": "json", "limit": 1}
    headers = {"User-Agent": "BestLocationFinder/1.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        result = response.json()
        if not result:
            return None, None
        return float(result[0]["lat"]), float(result[0]["lon"])
    except Exception:
        return None, None


def _make_geocoded_row(name, lat, lon):
    """Build a minimal Series row from geocoded coords for distance calculations."""
    return pd.Series({
        "location": name.title(),
        "landmark": name.title(),
        "state": "Unknown",
        "country": "Unknown",
        "lat_location": lat,
        "lon_location": lon,
        "ratings": "N/A",
        "total_reviews": "N/A",
        "final_score": 100,
    })


def fmt_rating(rating):
    """Format rating safely without coordinates."""
    try:
        if rating is None or rating == "" or rating == "NA":
            return "Not Rated"
        num = float(rating)
        if math.isnan(num):
            return "Not Rated"
        return f"⭐ {num:.1f}"
    except (ValueError, TypeError):
        return "Not Rated"


def fmt_reviews(reviews):
    """Format review count safely."""
    try:
        if reviews is None or reviews == "" or reviews == "NA":
            return "No reviews"
        num = int(float(reviews))
        return f"{num:,} reviews"
    except (ValueError, TypeError):
        return "No reviews"


def fmt_state_country(state, country):
    """Format location nicely."""
    if pd.isna(state) or state == "Unknown":
        return f"{country}" if pd.notna(country) else "Location Unknown"
    return f"{state}, {country}"


def fmt_distance(km):
    """Format distance in readable way."""
    if km is None or km == float("inf"):
        return ""
    if km < 1:
        return f"{km*1000:.0f}m"
    return f"{km:.1f} km"


def fmt_time(hrs):
    """Format travel time."""
    h, m = int(hrs), int((hrs % 1) * 60)
    return f"{h}h {m}m" if h else f"{m}m"


_ENTITY_NOISE = re.compile(
    r'\b(the|a|an|distance|between|from|to|tell|me|hey|find|what|is|how|far|where|show|give)\b',
    flags=re.IGNORECASE
)

def clean_entity(text):
    try:
        cleaned = _ENTITY_NOISE.sub('', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except Exception as e:
        return text


# ============================================================================
# GREETING & LOCATION RESOLUTION
# ============================================================================

def resolve_location(query, session):
    try:
        location_query = extract_location(query)

        if location_query:
            cleaned_text   = re.sub(r'[^\w\s]', '', location_query)
            location_query = " ".join(
                w for w in cleaned_text.split() if w.lower() not in _NOISE_WORDS
            ).strip()

        if not location_query and session.get("last_location"):
            location_query = session["last_location"]

        return location_query
    except Exception as e:
        return ""


def handle_greeting(query, session):
    try:
        session["last_intent"] = "greeting"
        session["state"]       = "greeting"

        q = query.lower().strip()
        if any(kw in q for kw in ("help", "start", "begin", "reset", "restart", "new chat")):
            return (
                "I can help you with:\n"
                "  • Places to visit\n"
                "  • Hotels & restaurants\n"
                "  • Distance between locations\n"
                "  • Full trip itineraries\n\n"
                "Where would you like to go?"
            )

        if session.get("last_location"):
            return f"Welcome back! Still planning around {session['last_location']}?"

        return random.choice(_GREET_RESPONSES)
    except Exception as e:
        return "Hello! How can I help you today?"


# ============================================================================
# HOTEL QUERY HANDLER
# ============================================================================

def handle_hotel_query(query, hotel_df, vectorizer, tfidf_matrix, session):
    try:
        location_query = resolve_location(query, session)
        if not location_query:
            session["awaiting"] = "hotel_location"
            return "Please specify a location to find hotels (e.g., 'hotels in Goa')."

        session["last_location"] = location_query
        session["last_intent"]   = "hotel"
        session["awaiting"]      = None

        processed_query = preprocess(location_query)
        query_vector    = vectorizer.transform([processed_query])
        similarities    = cosine_similarity(query_vector, tfidf_matrix).flatten()

        hotel_df = hotel_df.copy().reset_index(drop=True)
        hotel_df["tfidf_score"] = similarities[:len(hotel_df)] * 100

        hotel_df["fuzz_score"] = hotel_df.apply(
            lambda r: max(
                calculate_similarity(processed_query, r.get("proc_location") or ""),
                calculate_similarity(processed_query, str(r.get("state") or "").lower()),
                calculate_similarity(processed_query, str(r.get("landmark") or "").lower()),
            ), axis=1
        )

        hotel_df["similarity"] = hotel_df["tfidf_score"] * 0.3 + hotel_df["fuzz_score"] * 0.7

        rejected   = session.get("rejected_hotels", set())
        candidates = (
            hotel_df[
                (hotel_df["similarity"] > 10) &
                (~hotel_df["location"].isin(rejected))
            ]
            .sort_values("similarity", ascending=False)
            .copy()
        )

        if candidates.empty:
            return f"No hotels found near '{location_query}'. Try a different location."

        # Re-rank by geographic distance
        ref     = candidates.iloc[0]
        ref_lat = ref.get("lat_location")
        ref_lon = ref.get("lon_location")

        if pd.notna(ref_lat) and pd.notna(ref_lon):
            candidates["dist_km"] = candidates.apply(
                lambda r: haversine(ref_lat, ref_lon, r["lat_location"], r["lon_location"])
                if pd.notna(r["lat_location"]) and pd.notna(r["lon_location"]) else float("inf"),
                axis=1
            )
            max_dist = candidates["dist_km"].replace(float("inf"), 0).max()
            if max_dist > 0:
                candidates["dist_score"] = (1 - candidates["dist_km"] / max_dist) * 100
            else:
                candidates["dist_score"] = 100.0
            candidates["final_score"] = candidates["similarity"] * 0.6 + candidates["dist_score"] * 0.4
            candidates = candidates.sort_values("final_score", ascending=False)

        session["hotel_pool"]   = candidates
        session["hotel_offset"] = 5
        results = candidates.head(5)

        output = f"🏨 Hotels near {location_query.title()}\n"
        output += "=" * 50 + "\n\n"

        for i, (_, row) in enumerate(results.iterrows(), 1):
            output += f"{i}. {row['location'].title()}\n"
            output += f"   📍 Near {row['landmark']}\n"
            output += f"   📌 {fmt_state_country(row['state'], row['country'])}\n"
            output += f"   {fmt_rating(row['ratings'])} • {fmt_reviews(row['total_reviews'])}\n"
            
            if "dist_km" in row and row["dist_km"] not in (0, float("inf")):
                output += f"   📏 {fmt_distance(row['dist_km'])} away\n"
            output += "\n"

        if len(candidates) > 5:
            output += f"Showing {len(results)} of {len(candidates)} results. Ask to 'show more hotels'.\n"

        print(output)
        session["last_results"] = results
        return output

    except Exception as e:
        return "Error finding hotels. Please try again."


# ============================================================================
# RESTAURANT QUERY HANDLER
# ============================================================================

def handle_restaurant_query(query, restaurant_df, vectorizer, tfidf_matrix, session):
    try:
        location_query = resolve_location(query, session)
        if not location_query:
            session["awaiting"] = "restaurant_location"
            return "Please specify a location to find restaurants (e.g., 'restaurants in Mumbai')."

        session["last_location"] = location_query
        session["last_intent"]   = "restaurant"
        session["awaiting"]      = None

        processed_query = preprocess(location_query)
        query_vector    = vectorizer.transform([processed_query])
        similarities    = cosine_similarity(query_vector, tfidf_matrix).flatten()

        restaurant_df = restaurant_df.copy().reset_index(drop=True)
        restaurant_df["tfidf_score"] = similarities[:len(restaurant_df)] * 100

        restaurant_df["fuzz_score"] = restaurant_df.apply(
            lambda r: max(
                calculate_similarity(processed_query, r.get("proc_location") or ""),
                calculate_similarity(processed_query, str(r.get("state") or "").lower()),
                calculate_similarity(processed_query, str(r.get("landmark") or "").lower()),
            ), axis=1
        )

        restaurant_df["similarity"] = restaurant_df["tfidf_score"] * 0.3 + restaurant_df["fuzz_score"] * 0.7

        if restaurant_df["similarity"].max() < 20:
            return f"'{location_query}' not found. Try a city or landmark name."

        rejected   = session.get("rejected_rests", set())
        candidates = (
            restaurant_df[
                (restaurant_df["similarity"] > 10) &
                (~restaurant_df["location"].isin(rejected))
            ]
            .sort_values("similarity", ascending=False)
            .copy()
        )

        if candidates.empty:
            return f"No restaurants found near '{location_query}'."

        ref     = candidates.iloc[0]
        ref_lat = ref.get("lat_location")
        ref_lon = ref.get("lon_location")

        if pd.notna(ref_lat) and pd.notna(ref_lon):
            candidates["dist_km"] = candidates.apply(
                lambda r: haversine(ref_lat, ref_lon, r["lat_location"], r["lon_location"])
                if pd.notna(r["lat_location"]) and pd.notna(r["lon_location"]) else float("inf"),
                axis=1
            )
            max_dist = candidates["dist_km"].replace(float("inf"), 0).max()
            if max_dist > 0:
                candidates["dist_score"] = (1 - candidates["dist_km"] / max_dist) * 100
            else:
                candidates["dist_score"] = 100.0
            candidates["final_score"] = candidates["similarity"] * 0.6 + candidates["dist_score"] * 0.4
            candidates = candidates.sort_values("final_score", ascending=False)

        session["restaurant_pool"]   = candidates
        session["restaurant_offset"] = 5
        results = candidates.head(5)

        output = f"🍽️ Restaurants near {location_query.title()}\n"
        output += "=" * 50 + "\n\n"

        for i, (_, row) in enumerate(results.iterrows(), 1):
            output += f"{i}. {row['location'].title()}\n"
            output += f"   📍 Near {row['landmark']}\n"
            output += f"   📌 {fmt_state_country(row['state'], row['country'])}\n"
            output += f"   {fmt_rating(row['ratings'])} • {fmt_reviews(row['total_reviews'])}\n"
            
            if "dist_km" in row and row["dist_km"] not in (0, float("inf")):
                output += f"   📏 {fmt_distance(row['dist_km'])} away\n"
            output += "\n"

        if len(candidates) > 5:
            output += f"Showing {len(results)} of {len(candidates)} results. Ask to 'show more restaurants'.\n"

        print(output)
        session["last_results"] = results
        return output

    except Exception as e:
        return "Error finding restaurants. Please try again."


# ============================================================================
# DISTANCE QUERY HANDLER
# ============================================================================

def handle_distance_query(query, data, vectorizer, tfidf_matrix, session):
    try:
        entities = extract_entities_bio(query)

        if len(entities) < 2:
            q_lower = query.lower()
            for sep in (' and ', ' from ', ' to '):
                if sep in q_lower:
                    idx = q_lower.index(sep)
                    part_a = q_lower[:idx].strip()
                    part_b = q_lower[idx + len(sep):].strip()
                    loc_a = extract_location(part_a)
                    loc_b = extract_location(part_b)
                    if loc_a and loc_b and loc_a != loc_b:
                        entities = [loc_a, loc_b]
                        break

        if len(entities) < 2:
            if len(entities) == 1 and session.get("last_location"):
                entities = [session["last_location"], entities[0]]
            else:
                session["awaiting"] = "distance_locations"
                return "Please specify two locations. Try: 'distance between Delhi and Mumbai'"

        entity_a = clean_entity(entities[0])
        entity_b = clean_entity(entities[1])

        if not entity_a or not entity_b:
            return "Could not parse location names. Please try again."

        ranked_a = score_and_rank(preprocess(entity_a), data, vectorizer, tfidf_matrix)
        ranked_b = score_and_rank(preprocess(entity_b), data, vectorizer, tfidf_matrix)

        place_a = ranked_a.iloc[0]
        place_b = ranked_b.iloc[0]

        if place_a["final_score"] < 20:
            lat, lon = get_lat_lon(entity_a)
            if lat is None:
                return f"Could not locate '{entity_a}'. Try a different place name."
            save_geocoded_to_csv(entity_a, lat, lon)
            place_a = _make_geocoded_row(entity_a, lat, lon)

        if place_b["final_score"] < 20:
            lat, lon = get_lat_lon(entity_b)
            if lat is None:
                return f"Could not locate '{entity_b}'. Try a different place name."
            save_geocoded_to_csv(entity_b, lat, lon)
            place_b = _make_geocoded_row(entity_b, lat, lon)

        if pd.isna(place_a["lat_location"]) or pd.isna(place_a["lon_location"]):
            return f"No coordinates available for '{place_a['location']}'."
        if pd.isna(place_b["lat_location"]) or pd.isna(place_b["lon_location"]):
            return f"No coordinates available for '{place_b['location']}'."

        dist = haversine(
            place_a["lat_location"], place_a["lon_location"],
            place_b["lat_location"], place_b["lon_location"]
        )

        route_str = (
            f"Within {place_a['state']}"
            if place_a["state"] == place_b["state"]
            else f"{place_a['state']} ↔️ {place_b['state']}"
        )

        mid_lat = (place_a["lat_location"] + place_b["lat_location"]) / 2
        mid_lon = (place_a["lon_location"] + place_b["lon_location"]) / 2
        mid_row, _ = find_nearest_in_df(mid_lat, mid_lon, data)

        output = "\n📍 Distance & Route Information\n"
        output += "=" * 50 + "\n\n"
        output += f"From: {place_a['location'].title()}, {place_a['state']}\n"
        output += f"To:   {place_b['location'].title()}, {place_b['state']}\n\n"
        output += f"📏 Distance: {dist:.1f} km\n"
        output += f"🛣️ Route: {route_str}\n\n"
        output += "Travel Time Estimates:\n"
        output += f"  🚗 By Car:    {fmt_time(dist / 60)} (avg 60 km/h)\n"
        output += f"  🚂 By Train:  {fmt_time(dist / 100)} (avg 100 km/h)\n"
        output += f"  ✈️  By Flight: {fmt_time(dist / 800)} (avg 800 km/h)\n"
        
        if mid_row is not None:
            output += f"\n💡 Midpoint Stop: {mid_row['location'].title()}, {mid_row['state']}\n"

        print(output)
        result = {"place_a": place_a, "place_b": place_b, "distance_km": dist}
        session["last_place_a"] = place_a
        session["last_place_b"] = place_b
        session["last_results"] = result
        session["last_intent"]  = "distance_query"
        return result

    except Exception as e:
        return "Error calculating distance. Please try again."


# ============================================================================
# NEAREST LOCATION FINDER
# ============================================================================

def find_nearest_in_df(lat, lon, df, max_km=None, n=1, exclude=None):
    try:
        if pd.isna(lat) or pd.isna(lon):
            return (None, None) if n == 1 else []
        if df.empty:
            return (None, None) if n == 1 else []

        valid = df[df['lat_location'].notna() & df['lon_location'].notna()].copy()

        if exclude:
            valid = valid[~valid['location'].isin(exclude)]
        if valid.empty:
            return (None, None) if n == 1 else []

        # Vectorized haversine
        lat1  = math.radians(lat)
        lon1  = math.radians(lon)
        lats2 = np.radians(valid['lat_location'].values)
        lons2 = np.radians(valid['lon_location'].values)
        dlat  = lats2 - lat1
        dlon  = lons2 - lon1
        a     = np.sin(dlat / 2) ** 2 + math.cos(lat1) * np.cos(lats2) * np.sin(dlon / 2) ** 2
        dists = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        valid = valid.copy()
        valid['_dist'] = dists
        valid = valid[valid['_dist'] > 0]

        if max_km is not None:
            valid = valid[valid['_dist'] <= max_km]
        if valid.empty:
            return (None, None) if n == 1 else []

        valid = valid.nsmallest(n, '_dist')
        result = valid.drop(columns=['_dist'])

        if n == 1:
            idx = result.index[0]
            return result.loc[idx], valid.loc[idx, '_dist']
        return list(result.itertuples(index=False))
    except Exception as e:
        return (None, None) if n == 1 else []


# ============================================================================
# ITINERARY QUERY HANDLER
# ============================================================================

def handle_itinerary_query(query, data, hotel_df, restaurant_df, vectorizer, tfidf_matrix, session):
    try:
        location_query = extract_itinerary_location(query)
        if not location_query:
            session["awaiting"] = "itinerary_destination"
            return "Please specify a destination for your trip (e.g., 'plan a 3-day trip to Jaipur')."

        days   = extract_days(query) or session.get("trip_days") or 2
        budget = extract_budget(query)
        if budget is None:
            budget = session.get("preferences", {}).get("budget")

        session["last_location"] = location_query
        session["last_intent"]   = "build_itinerary"
        session["awaiting"]      = None
        session["trip_days"]     = days
        if budget is not None:
            session["preferences"]["budget"] = budget

        processed       = preprocess(location_query)
        ranked          = score_and_rank(processed, data, vectorizer, tfidf_matrix)
        rejected_places = session.get("rejected_places", set())
        filtered_data   = data

        if rejected_places:
            ranked        = ranked[~ranked['landmark'].isin(rejected_places)]
            filtered_data = data[~data['landmark'].isin(rejected_places)]
            if ranked.empty:
                return f"We ran out of new places to suggest in '{location_query}'!"

        best = ranked.iloc[0]

        if best["final_score"] < 36:
            lat, lon = get_lat_lon(location_query)
            if lat is not None:
                save_geocoded_to_csv(location_query, lat, lon)
                nearest, ndist = find_nearest_in_df(lat, lon, data[data['type'] == 'place'])
                if nearest is None:
                    nearest, ndist = find_nearest_in_df(lat, lon, data)
                if nearest is not None and ndist is not None and ndist <= 300:
                    ranked = score_and_rank(preprocess(nearest['location']), data, vectorizer, tfidf_matrix)
                    best = ranked.iloc[0]
                elif nearest is not None:
                    return f"No destinations in our database are near '{location_query}'. Nearest: {nearest['location']} (~{ndist:.0f} km away)."
            if best["final_score"] < 36:
                return f"'{location_query}' not found. Try a city or place name."

        state_places = ranked[(ranked['state'] == best['state']) & (ranked['type'] == 'place')]
        if state_places.empty:
            return f"No places to visit found in '{location_query}'."

        best_place    = state_places.iloc[0]
        stops_per_day = 2
        chain         = build_chain(best_place, filtered_data, max_stops=days * stops_per_day, min_distance=5, place_only=True)
        day_groups    = [chain[i:i + stops_per_day] for i in range(0, len(chain), stops_per_day)][:days]

        output = f"\n🗺️  {days}-Day Itinerary for {location_query.title()}\n"
        output += "=" * 50 + "\n"
        output += f"📌 State: {best_place['state']}, {best_place['country']}\n"
        
        if isinstance(budget, (int, float)):
            daily = budget / days
            output += f"💰 Budget: ₹{budget:,.0f} (~₹{daily:,.0f}/day)\n"
        elif isinstance(budget, str):
            output += f"💰 Budget: {budget.title()}\n"
        output += "\n"

        visited_hotels = set(session.get("rejected_hotels", set()))
        visited_rests  = set(session.get("rejected_rests",  set()))
        MAX_STAY_KM = 30
        session["generated_plan"] = {}

        for day_num, stops in enumerate(day_groups, 1):
            if not stops:
                continue

            day_memory = {"places": [], "hotel": None, "rest": None}
            output += f"\n📅 DAY {day_num}\n"
            output += "-" * 50 + "\n"

            for idx, stop in enumerate(stops, 1):
                day_memory["places"].append(stop['landmark'])
                output += f"\n✨ Stop {idx}: {stop['landmark'].title()}\n"
                output += f"   📍 {fmt_state_country(stop['state'], stop['country'])}\n"

            # Find centroid
            valid_stops = [s for s in stops if pd.notna(s['lat_location']) and pd.notna(s['lon_location'])]
            if valid_stops:
                ref_lat = sum(s['lat_location'] for s in valid_stops) / len(valid_stops)
                ref_lon = sum(s['lon_location'] for s in valid_stops) / len(valid_stops)
            else:
                ref_lat, ref_lon = stops[0]['lat_location'], stops[0]['lon_location']

            stop_location = stops[0]['location']
            stop_state    = stops[0]['state']

            # Hotel
            hotel_row, h_dist = find_nearest_in_df(
                ref_lat, ref_lon,
                hotel_df[hotel_df['landmark'] == stop_location],
                max_km=MAX_STAY_KM, exclude=visited_hotels
            )
            if hotel_row is None:
                hotel_row, h_dist = find_nearest_in_df(
                    ref_lat, ref_lon,
                    hotel_df[hotel_df['state'] == stop_state],
                    max_km=MAX_STAY_KM, exclude=visited_hotels
                )

            if hotel_row is not None:
                visited_hotels.add(hotel_row['location'])
                day_memory["hotel"] = hotel_row['location']
                output += f"\n🏨 Hotel: {hotel_row['location'].title()}\n"
                output += f"   {fmt_rating(hotel_row['ratings'])} • {fmt_reviews(hotel_row['total_reviews'])}\n"

            # Restaurant
            rest_row, r_dist = find_nearest_in_df(
                ref_lat, ref_lon,
                restaurant_df[restaurant_df['landmark'] == stop_location],
                max_km=MAX_STAY_KM, exclude=visited_rests
            )
            if rest_row is None:
                rest_row, r_dist = find_nearest_in_df(
                    ref_lat, ref_lon,
                    restaurant_df[restaurant_df['state'] == stop_state],
                    max_km=MAX_STAY_KM, exclude=visited_rests
                )

            if rest_row is not None:
                visited_rests.add(rest_row['location'])
                day_memory["rest"] = rest_row['location']
                output += f"\n🍽️  Restaurant: {rest_row['location'].title()}\n"
                output += f"   {fmt_rating(rest_row['ratings'])} • {fmt_reviews(rest_row['total_reviews'])}\n"

            session["generated_plan"][day_num] = day_memory

        output += "\n" + "=" * 50 + "\n"
        print(output)
        session["last_results"] = chain
        return chain

    except Exception as e:
        return "Error creating itinerary. Please try again."


# ============================================================================
# LOCATION QUERY HANDLER
# ============================================================================

def handle_location_query(query, data, vectorizer, tfidf_matrix, session):
    try:
        if not query:
            session["awaiting"] = "location_name"
            return None

        rejected_places = session.get("rejected_places", set())
        filtered_data   = data[~data['landmark'].isin(rejected_places)] if rejected_places else data

        processed = preprocess(query)
        ranked    = score_and_rank(processed, filtered_data, vectorizer, tfidf_matrix)
        best      = ranked.iloc[0]

        if best["final_score"] < 36:
            lat, lon = get_lat_lon(query)
            if lat is not None:
                save_geocoded_to_csv(query, lat, lon)
                nearest, ndist = find_nearest_in_df(lat, lon, filtered_data[filtered_data['type'] == 'place'])
                if nearest is None:
                    nearest, ndist = find_nearest_in_df(lat, lon, filtered_data)
                if nearest is not None and ndist is not None and ndist <= 300:
                    ranked = score_and_rank(preprocess(nearest['location']), filtered_data, vectorizer, tfidf_matrix)
                    best = ranked.iloc[0]
                elif nearest is not None:
                    return f"No destinations nearby '{query}'. Nearest: {nearest['location']} (~{ndist:.0f} km away)."
            if best["final_score"] < 36:
                return f"'{query}' not found. Try a city or place name."

        if best["final_score"] < 40:
            print(f"(Showing results for '{best['location']}' — closest match)")

        session["last_location"] = best["location"]
        session["last_intent"]   = "location"

        chain = build_chain(best, filtered_data, place_only=True)
        total_dist  = 0.0
        chain_with_dists = []

        output = f"\n📍 Nearby Places in {best['location'].title()}\n"
        output += "=" * 50 + "\n\n"

        for i, row in enumerate(chain):
            label = LABELS[i] if i < len(LABELS) else str(i + 1)
            seg_dist = None
            
            output += f"{label}. {row['location'].title()}\n"
            output += f"   📌 {fmt_state_country(row['state'], row['country'])}\n"
            
            if i > 0:
                prev = chain[i - 1]
                prev_lbl = LABELS[i - 1] if i - 1 < len(LABELS) else str(i)
                if pd.notna(prev["lat_location"]) and pd.notna(row["lat_location"]):
                    seg_dist    = haversine(
                        prev["lat_location"], prev["lon_location"],
                        row["lat_location"],  row["lon_location"]
                    )
                    total_dist += seg_dist
                    output += f"   (↓ {fmt_distance(seg_dist)} from Stop {prev_lbl})\n"
            output += "\n"
            chain_with_dists.append({**row, "dist_from_prev_km": seg_dist})

        if len(chain) > 1 and total_dist > 0:
            output += f"Total Route: {len(chain)} stops | {total_dist:.1f} km end-to-end\n"
        output += "=" * 50 + "\n"

        print(output)
        session["last_results"]       = chain_with_dists
        session["last_chain_dist_km"] = total_dist
        return chain

    except Exception as e:
        return "Error fetching locations. Please try again."