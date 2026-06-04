import sys
import os
import io
import contextlib

CODE2_DIR = os.path.join(os.path.dirname(__file__), "code 2")
if CODE2_DIR not in sys.path:
    sys.path.insert(0, CODE2_DIR)

from load_location_data import load_all_data
from detect_intent import (
    session,
    classify_query,
    update_intent,
    handle_itinerary_intent,
    handle_correction_followup,
    SLOT_QUESTIONS,
    fallback_response,
)
from handlers import (
    handle_hotel_query,
    handle_restaurant_query,
    handle_distance_query,
    handle_location_query,
    handle_itinerary_query,
    handle_greeting,
)

data, hotel_df, restaurant_df, landmark_df, vectorizer, tfidf_matrix = load_all_data()


@contextlib.contextmanager
def _capture():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def _run_itinerary():
    with _capture() as buf:
        handle_itinerary_query(
            session["last_location"],
            data, hotel_df, restaurant_df, vectorizer, tfidf_matrix,
        )
    return buf.getvalue().strip() or "Sorry, I couldn't generate an itinerary."


def _run_correction(correction):
    c_type = correction.get("type")
    day = correction.get("day")
    slot = correction.get("slot")
    
    # 1. Ensure ALL blacklists exist in memory
    if "rejected_hotels" not in session: session["rejected_hotels"] = set()
    if "rejected_rests" not in session: session["rejected_rests"] = set()
    if "rejected_places" not in session: session["rejected_places"] = set() # NEW!
        
    # 2. Look up what was generated for that day
    plan = session.get("generated_plan", {})
    
    if day and day in plan:
        day_data = plan[day]
        
        # Blacklist Hotels
        if slot == "hotel" and day_data.get("hotel"):
            session["rejected_hotels"].add(day_data["hotel"])
            
        # Blacklist Restaurants
        elif slot in ["dinner", "restaurant", "lunch", "eat"] and day_data.get("rest"):
            session["rejected_rests"].add(day_data["rest"])
            
        # Blacklist Activities / Places (Morning is index 0, Afternoon is index 1)
        elif slot in ["morning", "first place"] and len(day_data.get("places", [])) > 0:
            session["rejected_places"].add(day_data["places"][0])
            
        elif slot in ["afternoon", "second place", "activity", "place"] and len(day_data.get("places", [])) > 1:
            session["rejected_places"].add(day_data["places"][1])
            
        # Blacklist the whole day if they hate it!
        elif c_type == "dislike_day":
            if day_data.get("hotel"): session["rejected_hotels"].add(day_data["hotel"])
            if day_data.get("rest"): session["rejected_rests"].add(day_data["rest"])
            for p in day_data.get("places", []):
                session["rejected_places"].add(p)
            
    # 3. Clear the pending states
    session["pending_correction"] = None
    session["awaiting"] = None
    
    return _run_itinerary()


def get_response(query: str) -> str:
    query = query.strip()
    if not query:
        return "Please enter a question."

    # 1. Handle ITINERARY slots (e.g. asking for Budget)
    if isinstance(session.get("awaiting"), str) and session["awaiting"] in SLOT_QUESTIONS:
        question = handle_itinerary_intent(query)
        if question:
            return question
        return _run_itinerary()

    # 2. Route the query using our upgraded router
    kind, payload = classify_query(query)

    # 3. Handle CORRECTIONS using our new state machine
    if kind == "correction":
        # This handles both brand new corrections and follow-up answers automatically
        correction, question = handle_correction_followup(query)
        if question:
            return question
        
        # If no question is returned, the state machine is complete!
        return _run_correction(correction)

    # 4. Handle standard intents
    if kind == "intent":
        intent = update_intent(session.get("last_intent"), query)

        if intent == "greeting":
            with _capture() as buf:
                handle_greeting(query)
            return buf.getvalue().strip() or "Hello! How can I help you today?"

        if intent == "itinerary":
            question = handle_itinerary_intent(query)
            if question:
                return question
            return _run_itinerary()

        with _capture() as buf:
            if intent == "hotel":
                handle_hotel_query(query, data, hotel_df, vectorizer, tfidf_matrix)
            elif intent == "restaurant":
                handle_restaurant_query(query, data, restaurant_df, vectorizer, tfidf_matrix)
            elif intent == "distance":
                handle_distance_query(query, data, vectorizer, tfidf_matrix)
            elif intent == "location":
                hotel_words = ["hotel", "room", "stay", "accommodation", "lodge", "resort"]
                if any(w in query.lower() for w in hotel_words):
                    handle_hotel_query(query, data, hotel_df, vectorizer, tfidf_matrix)
                else:
                    handle_location_query(query, data, vectorizer, tfidf_matrix)
            elif intent == "taxicab":
                print(f"Finding taxis in {session.get('last_location', 'your area')}...")
            else:
                fallback_response()

        response = buf.getvalue().strip()
        return response or "Sorry, I couldn't find anything for that query."

    return "I can help with hotels, restaurants, taxis, and trip planning. What would you like?"