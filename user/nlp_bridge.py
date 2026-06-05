import sys
import os
import io
import contextlib

CODE2_DIR = os.path.join(os.path.dirname(__file__), "code 2")
if CODE2_DIR not in sys.path:
    sys.path.insert(0, CODE2_DIR)

from load_location_data import load_all_data
from query_utils import extract_itinerary_location, extract_days, extract_budget
from detect_intent import (
    session as _new_session,
    detect_intent,
    reset_session,
    get_missing_info,
    fill_slot,
    is_correction_query,
    handle_correction_followup,
)
from handlers import (
    handle_hotel_query,
    handle_restaurant_query,
    handle_distance_query,
    handle_location_query,
    handle_itinerary_query,
    handle_greeting,
)

# Load your data and ML models
data, hotel_df, restaurant_df, landmark_df, vectorizer, tfidf_matrix = load_all_data()

# Global session dictionary
session = _new_session()

@contextlib.contextmanager
def _capture():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf

def _run_itinerary(session_obj):
    """Executes the generator using the collected data."""
    with _capture() as buf:
        handle_itinerary_query(
            session_obj["last_location"],
            data, hotel_df, restaurant_df, vectorizer, tfidf_matrix,
            session_obj,
        )
    return buf.getvalue().strip() or "Sorry, I couldn't generate an itinerary."

def _run_correction(correction, session_obj):
    """Applies blacklists and regenerates the itinerary."""
    c_type = correction.get("type")
    day = correction.get("day")
    slot = correction.get("slot")

    if "rejected_hotels" not in session_obj: session_obj["rejected_hotels"] = set()
    if "rejected_rests" not in session_obj: session_obj["rejected_rests"] = set()
    if "rejected_places" not in session_obj: session_obj["rejected_places"] = set()

    plan = session_obj.get("generated_plan", {})

    if day and day in plan:
        day_data = plan[day]

        if slot == "hotel" and day_data.get("hotel"):
            session_obj["rejected_hotels"].add(day_data["hotel"])

        elif slot in ["dinner", "restaurant", "lunch", "eat"] and day_data.get("rest"):
            session_obj["rejected_rests"].add(day_data["rest"])

        elif slot in ["morning", "first place"] and len(day_data.get("places", [])) > 0:
            session_obj["rejected_places"].add(day_data["places"][0])

        elif slot in ["afternoon", "second place", "activity", "place"] and len(day_data.get("places", [])) > 1:
            session_obj["rejected_places"].add(day_data["places"][1])

        elif c_type == "dislike_day":
            if day_data.get("hotel"): session_obj["rejected_hotels"].add(day_data["hotel"])
            if day_data.get("rest"): session_obj["rejected_rests"].add(day_data["rest"])
            for p in day_data.get("places", []):
                session_obj["rejected_places"].add(p)

    session_obj["pending_correction"] = None
    session_obj["state"] = "reviewing"

    session_obj["last_location"] = session_obj["collected"]["destination"]
    session_obj["trip_days"] = session_obj["collected"]["days"]

    return _run_itinerary(session_obj)


def get_response(query: str, user_session=None) -> str:
    """The Main Brain: Routes requests based on conversation state."""
    _session = user_session if user_session is not None else session

    query = query.strip()
    if not query:
        return "Please enter a question."

    intent = detect_intent(query)
    current_state = _session.get("state", "greeting")

    # ---------------------------------------------------------
    # GLOBAL OVERRIDES (Can happen at any time)
    # ---------------------------------------------------------
    if intent == "goodbye":
        return "Goodbye! Hope to help you plan another trip soon. Safe travels!"

    if intent == "restart":
        reset_session(_session)
        return "Everything has been reset! Where would you like to travel?"

    if intent == "context_switch":
        _session["collected"]["destination"] = None
        _session["collected"]["days"] = None
        _session["last_location"] = None
        _session["trip_days"] = None
        _session["state"] = "collecting_info"

        new_loc = extract_itinerary_location(query)
        if new_loc:
            _session["collected"]["destination"] = new_loc.title()
            _session["last_location"] = new_loc.title()
        # Falls through to collecting_info so the bot asks the next missing detail

    # ---------------------------------------------------------
    # ONE-OFF DIRECT QUERIES (Handling distances, standalone hotels)
    # ---------------------------------------------------------
    # Only process these if we aren't actively trying to extract answers for an itinerary
    if (intent in ["hotel", "restaurant", "distance_query", "taxicab"]
            and current_state != "collecting_info"
            and not (current_state == "reviewing" and is_correction_query(query))):
        with _capture() as buf:
            if intent == "hotel":
                handle_hotel_query(query, hotel_df, vectorizer, tfidf_matrix, _session)
            elif intent == "restaurant":
                handle_restaurant_query(query, restaurant_df, vectorizer, tfidf_matrix, _session)
            elif intent == "distance_query":
                handle_distance_query(query, data, vectorizer, tfidf_matrix, _session)
            elif intent == "taxicab":
                print(f"Finding taxis in {_session.get('last_location', 'your area')}...")
        response = buf.getvalue().strip()
        return response or "Sorry, I couldn't find anything for that query."

    # ---------------------------------------------------------
    # STATE: PENDING CORRECTIONS (Active Follow-ups)
    # ---------------------------------------------------------
    if _session.get("pending_correction"):
        correction, question = handle_correction_followup(query, _session)
        if question:
            return question
        return _run_correction(correction, _session)

    # ---------------------------------------------------------
    # STATE 1: GREETING
    # ---------------------------------------------------------
    if current_state == "greeting":
        if intent in ["build_itinerary", "location"]:
            _session["state"] = "collecting_info"

            # Deep extract from initial sentence
            loc = extract_itinerary_location(query)
            if loc:
                _session["collected"]["destination"] = loc.title()
                _session["last_location"] = loc.title()

            days = extract_days(query)
            if days:
                _session["collected"]["days"] = days
                _session["trip_days"] = days

            budget = extract_budget(query)
            if budget:
                _session["collected"]["budget"] = budget
                _session["preferences"]["budget"] = budget

            # Fall through to "collecting_info" block to ask questions

        elif intent == "greeting":
            return handle_greeting(query, _session)
        else:
            return "I can help you plan an amazing trip! Just tell me where you want to go."

    # ---------------------------------------------------------
    # STATE 2: COLLECTING INFO
    # ---------------------------------------------------------
    if _session["state"] == "collecting_info":
        # Process the answer — extract clean location name when filling destination
        if intent in ["answer", "location", "unknown", "build_itinerary"]:
            missing = get_missing_info(_session)
            if missing:
                if missing == "destination":
                    loc = extract_itinerary_location(query)
                    if loc:
                        fill_slot("destination", loc, _session)
                    # if no location found, leave destination empty so we ask again
                else:
                    fill_slot(missing, query, _session)

        # Check what we still need to ask
        missing = get_missing_info(_session)

        if missing == "destination":
            return "Where would you like to travel?"
        elif missing == "days":
            return "How many days are you planning to stay?"
        else:
            _session["state"] = "reviewing"
            return _run_itinerary(_session)

    # ---------------------------------------------------------
    # STATE 3: REVIEWING ITINERARY
    # ---------------------------------------------------------
    if current_state == "reviewing":
        if intent == "refine_itinerary" or is_correction_query(query):
            correction, question = handle_correction_followup(query, _session)
            if question:
                return question
            return _run_correction(correction, _session)

        elif intent == "confirm":
            _session["state"] = "finalized"
            dest = _session['collected']['destination'] or "your destination"
            return f"Awesome! Have a wonderful trip to {dest}! Let me know if you want to plan another one."

    # ---------------------------------------------------------
    # STATE 4: FINALIZED — trip confirmed, offer to start fresh
    # ---------------------------------------------------------
    if current_state == "finalized":
        if intent in ["build_itinerary", "location", "greeting"]:
            reset_session(_session)
            _session["state"] = "collecting_info"
            # Try to extract info from the query itself so one-shot requests work
            loc = extract_itinerary_location(query)
            if loc:
                _session["collected"]["destination"] = loc.title()
                _session["last_location"] = loc.title()
            days = extract_days(query)
            if days:
                _session["collected"]["days"] = days
                _session["trip_days"] = days
            missing = get_missing_info(_session)
            if missing is None:
                _session["state"] = "reviewing"
                return _run_itinerary(_session)
            elif missing == "days":
                return f"Great! How many days are you planning for {_session['collected']['destination']}?"
            return "Let's plan your next trip! Which destination would you like to visit?"
        return "Your trip plan is saved! Say 'start over' to plan a new trip, or ask me about hotels, restaurants, or distances."

    # Fallback
    return "I am here to help you plan your trip. What would you like to do?"
