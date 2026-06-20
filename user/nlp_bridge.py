from .nlp_engine.load_location_data import load_all_data
from .nlp_engine.query_utils import extract_itinerary_location, extract_days, extract_budget
from .nlp_engine.detect_intent import (session as _new_session, detect_intent, reset_session, get_missing_info, fill_slot, is_correction_query, handle_correction_followup)
from .nlp_engine.handlers import (handle_hotel_query, handle_restaurant_query, handle_distance_query, handle_itinerary_query, handle_greeting)

data, hotel_df, restaurant_df, landmark_df, vectorizer, tfidf_matrix = load_all_data()

# global session used when no per-user session is passed in
session = _new_session()


def _run_itinerary(session_obj):
    return handle_itinerary_query(
        session_obj["last_location"],
        data, hotel_df, restaurant_df, vectorizer, tfidf_matrix,
        session_obj,
    ) or "Sorry, I couldn't generate an itinerary."


def _run_correction(correction, session_obj):
    """Applies the blacklists from a correction dict and regenerates the itinerary."""
    c_type = correction.get("type")
    day    = correction.get("day")
    slot   = correction.get("slot")

    if "rejected_hotels" not in session_obj: session_obj["rejected_hotels"] = set()
    if "rejected_rests"  not in session_obj: session_obj["rejected_rests"]  = set()
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
            # reject everything on that day at once
            if day_data.get("hotel"): session_obj["rejected_hotels"].add(day_data["hotel"])
            if day_data.get("rest"):  session_obj["rejected_rests"].add(day_data["rest"])
            for p in day_data.get("places", []):
                session_obj["rejected_places"].add(p)

    session_obj["pending_correction"] = None
    session_obj["state"] = "reviewing"
    session_obj["last_location"] = session_obj["collected"]["destination"]
    session_obj["trip_days"]     = session_obj["collected"]["days"]

    return _run_itinerary(session_obj)


def get_response(query: str, user_session=None) -> str:
    """Main entry point — routes every message through the right handler based on state."""
    _session = user_session if user_session is not None else session

    query = query.strip()
    if not query:
        return "Please enter a question."

    intent        = detect_intent(query)
    current_state = _session.get("state", "greeting")

    # --- global overrides, work in any state ---

    if intent == "goodbye":
        return "Goodbye! Hope to help you plan another trip soon. Safe travels!"

    if intent == "restart":
        reset_session(_session)
        return "Everything has been reset! Where would you like to travel?"

    if intent == "context_switch":
        _session["collected"]["destination"] = None
        _session["collected"]["days"]        = None
        _session["last_location"]            = None
        _session["trip_days"]                = None
        _session["state"]                    = "collecting_info"

        new_loc = extract_itinerary_location(query)
        if new_loc:
            _session["collected"]["destination"] = new_loc.title()
            _session["last_location"]            = new_loc.title()

    # --- standalone one-off queries (skip when we're mid-itinerary collection) ---

    if (intent in ["hotel", "restaurant", "distance_query", "taxicab"]
            and current_state != "collecting_info"
            and not (current_state == "reviewing" and is_correction_query(query))):
        response = ""
        if intent == "hotel":
            response = handle_hotel_query(query, hotel_df, vectorizer, tfidf_matrix, _session)
        elif intent == "restaurant":
            response = handle_restaurant_query(query, restaurant_df, vectorizer, tfidf_matrix, _session)
        elif intent == "distance_query":
            response = handle_distance_query(query, data, vectorizer, tfidf_matrix, _session)
        elif intent == "taxicab":
            response = f"Finding taxis in {_session.get('last_location', 'your area')}..."
        return response or "Sorry, I couldn't find anything for that query."

    # --- pending multi-turn correction ---

    if _session.get("pending_correction"):
        correction, question = handle_correction_followup(query, _session)
        if question:
            return question
        return _run_correction(correction, _session)

    # --- state: greeting ---

    if current_state == "greeting":
        if intent in ["build_itinerary", "location"]:
            _session["state"] = "collecting_info"

            loc = extract_itinerary_location(query)
            if loc:
                _session["collected"]["destination"] = loc.title()
                _session["last_location"]            = loc.title()

            days = extract_days(query)
            if days:
                _session["collected"]["days"] = days
                _session["trip_days"]         = days

            budget = extract_budget(query)
            if budget:
                _session["collected"]["budget"]     = budget
                _session["preferences"]["budget"]   = budget

        elif intent == "greeting":
            return handle_greeting(query, _session)
        else:
            return "I can help you plan an amazing trip! Just tell me where you want to go."

    # --- state: collecting_info ---

    if _session["state"] == "collecting_info":
        if intent in ["answer", "location", "unknown", "build_itinerary"]:
            missing = get_missing_info(_session)
            if missing:
                if missing == "destination":
                    loc = extract_itinerary_location(query)
                    if loc:
                        fill_slot("destination", loc, _session)
                else:
                    fill_slot(missing, query, _session)

        missing = get_missing_info(_session)

        if missing == "destination":
            return "Where would you like to travel?"
        elif missing == "days":
            return "How many days are you planning to stay?"
        else:
            _session["state"] = "reviewing"
            return _run_itinerary(_session)

    # --- state: reviewing ---

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

        elif intent == "build_itinerary":
            new_loc  = extract_itinerary_location(query)
            new_days = extract_days(query)
            new_bud  = extract_budget(query)
            _session["collected"]["destination"] = new_loc.title() if new_loc else None
            _session["last_location"]            = new_loc.title() if new_loc else None
            _session["collected"]["days"]        = new_days
            _session["trip_days"]                = new_days
            if new_bud:
                _session["collected"]["budget"]   = new_bud
                _session["preferences"]["budget"] = new_bud
            _session["state"] = "collecting_info"
            if not new_loc:
                return "Where would you like to travel?"
            if not new_days:
                return "How many days are you planning?"
            _session["state"] = "reviewing"
            return _run_itinerary(_session)

    # --- state: finalized ---

    if current_state == "finalized":
        if intent in ["build_itinerary", "location", "greeting"]:
            reset_session(_session)
            _session["state"] = "collecting_info"
            loc = extract_itinerary_location(query)
            if loc:
                _session["collected"]["destination"] = loc.title()
                _session["last_location"]            = loc.title()
            days = extract_days(query)
            if days:
                _session["collected"]["days"] = days
                _session["trip_days"]         = days
            missing = get_missing_info(_session)
            if missing is None:
                _session["state"] = "reviewing"
                return _run_itinerary(_session)
            elif missing == "days":
                return f"Great! How many days are you planning for {_session['collected']['destination']}?"
            return "Let's plan your next trip! Which destination would you like to visit?"
        return "Your trip plan is saved! Say 'start over' to plan a new trip, or ask me about hotels, restaurants, or distances."

    return "I am here to help you plan your trip. What would you like to do?"
