import re

session = {
    "last_location":  None,
    "last_intent":    None,
    "last_results":   [],
    "trip_days":      None,
    "preferences":    {"veg_only": False, "budget": None},
    "visited_places": [],
    "disliked_days":  [],
    "awaiting":       None,
    "pending_correction": None,
}

def reset_session():
    try:
        session["last_location"]  = None
        session["trip_days"]      = None
        session["preferences"]    = {"veg_only": False, "budget": None}
        session["last_results"]   = []
        session["visited_places"] = []
        session["disliked_days"]  = []
        session["last_intent"]    = None
        session["awaiting"]       = None
        session["pending_correction"] = None
    except Exception as e:
        print(f"Error in reset_session: {e}")


intents = {
    "greeting-keywords":       {"keywords": [
        'hi', 'hello', 'hey', 'hiya', 'yo', 'sup', 'greetings', 'howdy',
        'namaste', 'namaskar', 'salaam', 'good morning', 'good afternoon',
        'good evening', 'good night', 'how are you', "what's up",
        'hy', 'hii', 'helo', 'heya', 'hai',
    ]},
    "hotel-keywords":          {"keywords": ['hotel', 'room', 'stay', 'accommodation', 'lodge', 'resort', 'book']},
    "location-keywords":       {"keywords": ['nearby', 'near', 'close to', 'around', 'some nearby', 'where i can stay', 'i can stay', 'can stay']},
    "distance-query-keywords": {"keywords": ['distance', 'how far', 'km', 'miles', 'haversine']},
    "restaurant-keywords":     {"keywords": ['restaurant', 'food', 'dining', 'eatery', 'cafe', 'diner', 'bistro', 'eat']},
    "taxicab-keywords":        {"keywords": ['taxi', 'cab', 'ride', 'transport', 'uber', 'lyft', 'taxicab', 'chauffeur']},
    "itinerary-keywords":      {"keywords": ['itinerary', 'trip plan', 'travel plan', 'plan my trip', 'plan a trip', 'day tour', 'day trip']},
}

INTENT_PRIORITY = ['location', 'restaurant', 'taxicab', 'hotel', 'distance', 'itinerary']

_WORD_NUMS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
_ITIN_KWS  = ['itinerary', 'trip plan', 'travel plan', 'plan my trip', 'plan a trip']
_DIST_KWS  = ['distance', 'how far', 'km', 'miles', 'haversine']


def detect_intent(query):
    try:
        q = query.lower()

        if any(re.search(r'\b' + re.escape(kw) + r'\b', q) for kw in _DIST_KWS):
            return 'distance'

        has_days = re.search(r'\b\d+\s*(?:day|days|night|nights)\b', q) or \
                   any(re.search(rf'\b{w}\s+(?:day|days|night|nights)\b', q) for w in _WORD_NUMS)
        if has_days or any(kw in q for kw in _ITIN_KWS):
            return 'itinerary'

        matched = []
        for key, details in intents.items():
            name = key.replace('-keywords', '')
            for kw in details['keywords']:
                # ADDED s? HERE: This allows 'restaurant' to match 'restaurants'
                if re.search(rf'\b{re.escape(kw)}s?\b', q):
                    matched.append(name)
                    break

        if not matched:
            return None

        for intent in reversed(INTENT_PRIORITY):
            if intent in matched:
                return intent

        return matched[0]
    except Exception as e:
        print(f"Error in detect_intent: {e}")
        return None


def update_intent(current_intent, query):
    try:
        new_intent = detect_intent(query)

        if new_intent is None:
            return current_intent

        if new_intent == 'location' and current_intent not in ('location', None, 'greeting'):
            return current_intent

        if new_intent == 'greeting':
            reset_session()
            session["last_intent"] = 'greeting'
            return 'greeting'

        session["last_intent"] = new_intent
        return new_intent
    except Exception as e:
        print(f"Error in update_intent: {e}")
        return current_intent


def add_intent(name, keywords):
    try:
        key = f"{name}-keywords"
        intents[key] = {"keywords": list(keywords)}
        if name not in INTENT_PRIORITY:
            INTENT_PRIORITY.append(name)
    except Exception as e:
        print(f"Error in add_intent: {e}")


def fallback_response():
    try:
        print("I can help with hotels, restaurants, taxis, and trip planning. What would you like?")
    except Exception as e:
        print(f"Error in fallback_response: {e}")


DAY_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "1st":   1, "2nd":    2, "3rd":   3, "4th":    4, "5th":   5,
}

SLOT_KEYWORDS = {
    "morning":   ["morning", "first place", "first spot"],
    "afternoon": ["afternoon", "second place", "activity", "place", "spot"], # Added "place" & "spot"
    "dinner":    ["dinner", "restaurant", "food", "lunch", "eat", "restaraunt", "restarant"], # Added typos
    "hotel":     ["hotel", "stay", "accommodation", "lodge", "resort"],
}

VISITED_KEYWORDS = [
    "already visited", "already been", "been there",
    "visited before", "already seen", "i've visited",
    "i have visited", "already went", "went there",
]

DISLIKE_KEYWORDS = [
    "don't like", "dont like", "not good", "boring",
    "bad plan", "redo", "new plan", "different plan",
    "change it", "change the plan", "change day",
    "not happy", "redo this",
]

CHANGE_SLOT_KEYWORDS = [
    "change the", "replace the", "different", "another",
    "suggest another", "give me another", "swap","switch", "change", "instead of", "alternative to", "other than", "new", "not the", "not this", "not that",
]


def extract_day_number(q):
    try:
        m = re.search(r'\bday\s*(\d+)\b', q) or re.search(r'\b(\d+)(?:st|nd|rd|th)?\s*day\b', q)
        if m:
            return int(m.group(1))
        for word, num in DAY_WORDS.items():
            if re.search(rf'\b{re.escape(word)}\b', q):
                return num
        return None
    except Exception as e:
        print(f"Error in extract_day_number: {e}")
        return None


def extract_slot(q):
    try:
        for slot, keywords in SLOT_KEYWORDS.items():
            for kw in keywords:
                # ADDED s?: Now it matches "restaurant", "restaurants", "restaraunts", etc.
                if re.search(rf'\b{re.escape(kw)}s?\b', q):
                    return slot
        return None
    except Exception as e:
        print(f"Error in extract_slot: {e}")
        return None


def extract_correction_type(q):
    try:
        def has_match(kws):
            # Added s? here too, just in case!
            return any(re.search(rf'\b{re.escape(kw)}s?\b', q) for kw in kws)

        if has_match(VISITED_KEYWORDS):
            return "visited"
        
        slot = extract_slot(q)
        
        # UPGRADED: If they use a word like "switch", we ALWAYS lock it in as a change intent!
        # If the slot is missing, the state machine will gracefully ask them "Which part?"
        if has_match(CHANGE_SLOT_KEYWORDS):
            return "change_slot"
            
        if has_match(DISLIKE_KEYWORDS):
            return "change_slot" if slot else "dislike_day"
            
        return None
    except Exception as e:
        print(f"Error in extract_correction_type: {e}")
        return None


def is_correction_query(query):
    try:
        q = query.lower()
        def has_match(kws):
            return any(re.search(rf'\b{re.escape(kw)}\b', q) for kw in kws)
        
        # We removed the 'has_day_ref' requirement. 
        # If they use a change keyword, we catch it, so we can ask them for the day!
        return (
            has_match(VISITED_KEYWORDS) or
            has_match(DISLIKE_KEYWORDS) or
            has_match(CHANGE_SLOT_KEYWORDS)
        )
    except Exception as e:
        print(f"Error in is_correction_query: {e}")
        return False


def parse_correction(query):
    try:
        q = query.lower()
        return {
            "type": extract_correction_type(q),
            "day":  extract_day_number(q),
            "slot": extract_slot(q),
        }
    except Exception as e:
        print(f"Error in parse_correction: {e}")
        return {"type": None, "day": None, "slot": None}


def classify_query(query):
    try:
        # 1. Catch active follow-ups
        if session.get("pending_correction"):
            return ("correction", session.get("pending_correction"))

        # 2. Safely check if we have previous results (works for both Lists and DataFrames)
        last_res = session.get("last_results")
        has_results = last_res is not None and len(last_res) > 0
        
        # 3. Process brand new correction requests
        if has_results and is_correction_query(query):
            return ("correction", parse_correction(query))
        
        # 4. Standard intents
        intent = detect_intent(query)
        if intent:
            return ("intent", intent)
            
        return ("unknown", None)
    except Exception as e:
        print(f"Error in classify_query: {e}")
        return ("unknown", None)


SLOT_QUESTIONS = {
    "last_location":   "Which city or destination are you planning to visit?",
    "trip_days":       "How many days is your trip?",
    "budget":          "What is your budget? (e.g. 3000 or budget / mid-range / luxury)",
    "veg_only":        "Do you prefer vegetarian food only? (yes / no)",
    "correction_day":  "Which day would you like me to change?",
    "correction_slot": "Which part — morning, afternoon, dinner, or hotel?",
}

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def get_missing_slot():
    try:
        if not session["last_location"]:
            return "last_location"
        if not session["trip_days"]:
            return "trip_days"
        if session["preferences"]["budget"] is None:
            return "budget"
        if session["preferences"]["veg_only"] is None:
            return "veg_only"
        return None
    except Exception as e:
        print(f"Error in get_missing_slot: {e}")
        return None


def fill_slot(slot, raw_answer):
    try:
        answer = raw_answer.strip().lower()

        if slot == "last_location":
            session["last_location"] = raw_answer.strip().title()
            return True

        if slot == "trip_days":
            match = re.search(r'\b(\d+)\b', answer)
            if match:
                session["trip_days"] = int(match.group(1))
                return True
            for word, num in WORD_TO_NUM.items():
                if re.search(rf'\b{word}\b', answer):
                    session["trip_days"] = num
                    return True
            return False

        if slot == "budget":
            amount_match = re.search(r'([\d,]+)', answer)
            if amount_match:
                amount = float(amount_match.group(1).replace(',', ''))
                session["preferences"]["budget"] = amount
                return True

            if any(w in answer for w in ["budget", "cheap", "low", "economic"]):
                session["preferences"]["budget"] = "budget"
            elif any(w in answer for w in ["luxury", "high", "premium", "expensive"]):
                session["preferences"]["budget"] = "luxury"
            else:
                session["preferences"]["budget"] = "mid-range"
            return True

        if slot == "veg_only":
            session["preferences"]["veg_only"] = answer in (
                "yes", "y", "veg", "vegetarian", "pure veg", "only veg"
            )
            return True

        return False
    except Exception as e:
        print(f"Error in fill_slot: {e}")
        return False


def _extract_slots_from_query(query):
    try:
        q = query.lower()

        loc_match = re.search(
            r'\b(?:to|in|for|visit|at|near)\s+(?!(?:spend|plan|want|show|give|book|stay|travel)\b)([A-Za-z][a-z]+(?:\s[A-Za-z][a-z]+)?)',
            q
        )
        if loc_match and not session["last_location"]:
            session["last_location"] = loc_match.group(1).strip().title()
        elif not session["last_location"]:
            direct_loc = re.search(
                r'\b(?:new delhi|delhi|jaipur|agra|mumbai|bangalore|kolkata|chennai|hyderabad|pune|lucknow|varanasi|taj mahal|taj)\b',
                q
            )
            if direct_loc:
                session["last_location"] = direct_loc.group(0).strip().title()

        day_match = re.search(r'\b(\d+)\s*(?:day|days|night|nights)\b', q)
        if day_match and not session["trip_days"]:
            session["trip_days"] = int(day_match.group(1))
        else:
            for word, num in WORD_TO_NUM.items():
                if re.search(rf'\b{word}\s+(?:day|days|night|nights)\b', q):
                    if not session["trip_days"]:
                        session["trip_days"] = num
                    break

        if "veg" in q or "vegetarian" in q:
            session["preferences"]["veg_only"] = True

        if any(w in q for w in ["budget", "cheap", "economic"]):
            session["preferences"]["budget"] = "budget"
        elif any(w in q for w in ["luxury", "premium", "expensive"]):
            session["preferences"]["budget"] = "luxury"
    except Exception as e:
        print(f"Error in _extract_slots_from_query: {e}")


def handle_itinerary_intent(user_input):
    try:
        if session["awaiting"] and isinstance(session["awaiting"], str) \
                and session["awaiting"] in SLOT_QUESTIONS:
            slot   = session["awaiting"]
            filled = fill_slot(slot, user_input)
            if not filled:
                return SLOT_QUESTIONS[slot]
            session["awaiting"] = None
        else:
            _extract_slots_from_query(user_input)

        missing = get_missing_slot()
        if missing:
            session["awaiting"] = missing
            return SLOT_QUESTIONS[missing]

        return None
    except Exception as e:
        print(f"Error in handle_itinerary_intent: {e}")
        return None


def handle_correction_followup(user_input):
    try:
        q = user_input.lower()
        
        correction: dict = session.get("pending_correction") or parse_correction(user_input) or {}

        # 2. Check if we need a day
        if correction.get("day") is None:
            day = extract_day_number(q)
            if day is None:
                # Save state and ask for the day
                session["pending_correction"] = correction
                return None, SLOT_QUESTIONS["correction_day"]
            correction["day"] = day

        # 3. Check if we need a slot (if applicable)
        if correction.get("type") in ("change_slot", "visited") and correction.get("slot") is None:
            slot = extract_slot(q)
            if slot is None:
                # Save state and ask for the slot
                session["pending_correction"] = correction
                return None, SLOT_QUESTIONS["correction_slot"]
            correction["slot"] = slot

        # 4. If we have everything, clear the pending correction and proceed
        session["pending_correction"] = None 
        return correction, None
        
    except Exception as e:
        print(f"Error in handle_correction_followup: {e}")
        return session.get("pending_correction"), None