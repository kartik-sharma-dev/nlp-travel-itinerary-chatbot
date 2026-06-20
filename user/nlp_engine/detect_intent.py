import re

_kw_cache: dict = {}

def _kw_matches(kw: str, q: str) -> bool:
    # short keywords need full word boundaries so "hi" doesn't match "him",
    # longer ones only need a left boundary so "hotel" still matches "hotels"
    if kw not in _kw_cache:
        if len(kw) <= 3:
            _kw_cache[kw] = re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
        else:
            _kw_cache[kw] = re.compile(rf'\b{re.escape(kw)}', re.IGNORECASE)
    return bool(_kw_cache[kw].search(q))


def session():
    return {
        "state":    "greeting",
        "attempts": 0,
        "itinerary": None,
        "pending_correction": None,
        "awaiting": None,
        "collected": {
            "destination": None,
            "days":        None,
            "group_type":  None,
            "budget":      None,
            "interests":   None,
            "pace":        None,
        },
        "last_location":  None,
        "last_intent":    None,
        "last_results":   [],
        "trip_days":      None,
        "preferences":    {"veg_only": False, "budget": None},
        "visited_places": [],
        "disliked_days":  [],
        "rejected_hotels": set(),
        "rejected_rests":  set(),
        "rejected_places": set(),
        "generated_plan":  {},
    }


def reset_session(session):
    try:
        session["state"]    = "greeting"
        session["attempts"] = 0
        session["itinerary"] = None
        session["pending_correction"] = None
        session["awaiting"] = None
        session["last_location"]  = None
        session["last_intent"]    = None
        session["last_results"]   = []
        session["trip_days"]      = None
        session["preferences"]    = {"veg_only": False, "budget": None}
        session["visited_places"] = []
        session["disliked_days"]  = []
        session["rejected_hotels"] = set()
        session["rejected_rests"]  = set()
        session["rejected_places"] = set()
        session["generated_plan"]  = {}
        for key in session["collected"]:
            session["collected"][key] = None
    except Exception as e:
        print(f"Error in reset_session: {e}")


INTENTS = {
    "greeting": [
        "hi", "hello", "hey", "hiya", "yo", "sup", "greetings", "howdy",
        "good morning", "good afternoon", "good evening", "good night", "good day",
        "how are you", "what's up", "whats up", "wassup", "wazzup",
        "how's it going", "how are u", "what's good", "how goes it",
        "hy", "hii", "hiii", "helo", "heya", "heyy", "helloo",
        "hey there", "hi there", "hello there", "yo yo", "hola", "ello",
        "start", "begin", "reset", "restart", "new chat", "help me"
    ],
    "hotel": [
        "hotel", "room", "stay", "accommodation", "lodge", "resort", "book",
        "hostel", "motel", "inn", "airbnb", "guest house", "guesthouse",
        "bed and breakfast", "b&b", "dormitory",
        "where to stay", "place to stay", "places to stay",
        "overnight", "check in", "check out", "checkout",
        "booking", "reserve", "reservation",
        "rent a room", "need a room", "find a room",
        "place to crash", "crash",
        "budget hotel", "luxury hotel", "cheap stay",
        "affordable stay", "rooms available", "vacancy", "suite"
    ],
    "location": [
        "nearby", "near", "close to", "around", "around me",
        "in the area", "some nearby",
        "places to visit", "things to do", "what to do", "what to see",
        "tourist spots", "tourist places", "attractions", "sightseeing",
        "landmarks", "must see", "must visit", "points of interest",
        "popular spots", "famous places", "top places", "best places",
        "hidden gems", "off the beaten path",
        "explore", "visit", "go to", "show me", "find me",
        "suggest places", "recommend places",
        "where to go", "where can i go",
        "monument", "temple", "museum", "park",
        "fort", "palace", "garden", "market",
        "beach", "hill", "waterfall", "lake"
    ],
    "distance_query": [
        "distance", "how far", "km", "miles", "haversine",
        "how long", "travel time", "drive time",
        "walking distance", "time to reach",
        "how long to reach", "how do i reach",
        "how to reach", "far is",
        "distance between", "how many km",
        "how many miles", "route", "directions",
        "navigate", "way to get", "path to",
        "shortest route", "fastest route"
    ],
    "restaurant": [
        "restaurant", "food", "dining", "eatery", "cafe",
        "diner", "bistro", "eat",
        "restaraunt", "resturant", "reasturant", "restaurent",
        "hungry", "starving", "famished",
        "i am hungry", "so hungry",
        "lunch", "dinner", "breakfast",
        "brunch", "snack", "meal", "supper",
        "where to eat", "place to eat",
        "bite to eat", "something to eat",
        "good food", "grub", "munchies",
        "street food", "fast food",
        "takeaway", "takeout", "delivery",
        "vegetarian food", "veg food",
        "non veg", "cuisine",
        "bar", "pub", "lounge",
        "food court", "food stall"
    ],
    "taxicab": [
        "taxi", "cab", "ride", "transport",
        "uber", "lyft", "taxicab", "chauffeur",
        "bike taxi",
        "bus", "metro", "local train",
        "subway", "train", "tram",
        "drop me", "pick me up", "pickup",
        "commute", "get there", "get to",
        "book a ride", "hire a cab",
        "hire a taxi", "need a cab",
        "need a ride", "how to get",
        "how do i get", "take me to",
        "driver", "vehicle", "conveyance",
        "travel to", "go from", "reach there"
    ],
    "build_itinerary": [
        "plan a trip", "plan my trip", "plan me a trip",
        "trip to", "i am planning a trip",
        "i want to plan a trip", "i want to go to", "i want to visit",
        "i am going to", "make an itinerary for me",
        "help me plan my holiday", "travel plan",
        "create a travel plan", "create a trip", "make a trip",
        "plan a holiday", "plan my holiday", "holiday trip", "holiday plan",
        "plan a vacation", "plan my vacation", "vacation plan",
        "i need a travel schedule", "suggest a trip for me",
        "plan something for this winter", "build me a travel plan",
        "can you make me an itinerary", "help me figure out my trip",
        "suggest an itinerary", "plan a 5 day trip for me",
        "i want a week long holiday", "give me a travel plan",
        "create a holiday plan for my family", "i need an itinerary for 7 days",
        "plan a budget trip for me", "help me with my travel plans",
        "can you organise my trip", "i need a day by day plan",
        "put together a trip for me",
        "weekend trip", "weekend plan", "solo trip", "family trip",
        "group trip", "honeymoon plan", "budget trip", "backpacking",
        "travel guide", "tour guide", "tour plan", "complete guide",
        "travel schedule", "visit plan", "sightseeing plan"
    ],
    "answer": [
        "5 days", "one week", "two weeks",
        "around 10 days", "long weekend",
        "solo", "couple", "family of four",
        "group of friends", "college friends",
        "just me", "me and my wife",
        "budget", "mid range", "luxury",
        "nature and adventure",
        "shopping and sightseeing",
        "relaxed pace", "packed schedule",
        "i love food and culture"
    ],
    "refine_itinerary": [
        "can you swap day 2", "remove the museum", "change day 3",
        "i don't want that activity", "replace the restaurant",
        "make it more relaxed", "add one more day",
        "we don't like shopping", "add more food activities",
        "remove the adventure stuff", "make day 1 less packed",
        "i want more cultural activities", "change the pace",
        "redo day 2", "replace that attraction",
        "add more nature activities", "make it more budget friendly",
        "remove nightlife from the plan", "add a beach day",
        "swap the lunch restaurant", "adjust the schedule",
        "fewer activities per day", "remove the temple visit",
        "swap day 1 and day 2", "make it more romantic",
        "day 2 is not good", "i don't like this plan",
        "not happy with the itinerary", "this plan is not good",
        "day 3 needs changes", "i want to modify day 2"
    ],
    "context_switch": [
        "change destination", "switch destinations", "i changed my mind",
        "forget the previous plan", "let's go somewhere else",
        "i want to change to a different city",
        "actually i prefer a hill station", "switch to a beach destination",
        "somewhere in south india", "somewhere different",
        "what about instead",
        "i am in", "i'm in", "i am at", "i'm at",
        "currently in", "we are in", "we're in",
        "planning to visit", "want to explore",
    ],
    "restart": [
        "start again", "start over", "start fresh",
        "reset everything", "forget everything", "redo this",
        "begin again", "clear everything",
        "restart the planning", "try again from scratch"
    ],
    "confirm": [
        "yes", "yeah", "yep", "sure", "correct",
        "that's right", "sounds good", "perfect",
        "that looks great", "yes please", "absolutely",
        "definitely", "that's fine", "ok", "okay",
        "great", "love it"
    ],
    "deny": [
        "no", "nope", "not really", "that's wrong", "that's not right",
        "i don't want that", "no thank you", "not what i meant",
        "that's incorrect", "i didn't mean that"
    ],
    "goodbye": [
        "bye", "goodbye", "see you", "thanks bye",
        "i'm done", "thank you goodbye", "cheers", "take care",
        "that will do thanks", "thanks for your help",
        "i have everything i need"
    ],
    "out_of_scope": [
        "tell me a joke", "what is the weather today",
        "who won the world cup", "what is the capital of france",
        "can you write code for me", "what is the meaning of life",
        "tell me about history", "what is today's date",
        "recommend a movie", "what is the stock price",
        "tell me something interesting", "what is machine learning",
        "help me with math", "who is the prime minister",
        "what is the best phone to buy", "help me with my homework",
        "what is cricket score"
    ]
}


WORD_NUMS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
ITIN_KWS  = [
    'itinerary', 'trip plan', 'travel plan', 'plan my trip', 'plan a trip',
    'day tour', 'day trip', 'travel guide', 'tour plan', 'plan my day',
    'plan my visit', 'plan my journey', 'sightseeing plan', 'travel schedule',
]
DIST_KWS  = [
    'distance', 'how far', 'km', 'miles', 'haversine',
    'how long', 'travel time', 'drive time', 'walking distance',
    'time to reach', 'how to reach', 'how do i reach', 'distance between',
    'route', 'directions', 'navigate',
]

# restart must come before greeting so "restart the planning" doesn't match greeting's "restart" keyword
PRIORITY_ORDER = [
    "restart",
    "goodbye",
    "greeting",
    "hotel",
    "restaurant",       # before "location" — "go to a restaurant" must not match location's "go to"
    "distance_query",
    "taxicab",
    "build_itinerary",
    "refine_itinerary",
    "context_switch",
    "confirm",
    "deny",
    "location",         # broad catch-all, must stay last among the named intents
    "answer",
    "out_of_scope",
]

_BUILD_ITINERARY_REGEX = [
    re.compile(r'\bplan\b.{0,30}\btrip\b',      re.IGNORECASE),
    re.compile(r'\bplan\b.{0,30}\bholiday\b',   re.IGNORECASE),
    re.compile(r'\bplan\b.{0,30}\bvacation\b',  re.IGNORECASE),
    re.compile(r'\b\d+\s*day\b.{0,20}\btrip\b', re.IGNORECASE),
    re.compile(r'\btrip\b.{0,10}\bto\b',         re.IGNORECASE),
]


def detect_intent(query):
    try:
        q = query.lower().strip()
        for intent_name in PRIORITY_ORDER:
            keywords = INTENTS.get(intent_name, [])
            if any(_kw_matches(kw.lower(), q) for kw in keywords):
                return intent_name
        # regex fallback catches patterns like "plan a relaxing trip" where words are in the middle
        if any(p.search(q) for p in _BUILD_ITINERARY_REGEX):
            return "build_itinerary"
        return "unknown"
    except Exception as e:
        print(f"Error in detect_intent: {e}")
        return "unknown"


def get_missing_info(session):
    try:
        collected = session["collected"]
        if not collected.get("destination"): return "destination"
        if not collected.get("days"):        return "days"
        return None
    except Exception as e:
        print(f"Error in get_missing_info: {e}")
        return None


DAY_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "1st":   1, "2nd":    2, "3rd":   3, "4th":    4, "5th":   5,
}

SLOT_KEYWORDS = {
    "morning": [
        r"\bmorning[s]?\b", r"\ba\.?m\.?\b", r"\bearly\b", r"\bbreakfast\b",
        r"\bbrunch\b", r"\bfirst (?:place|spot|thing)\b", r"\bsunrise\b",
        r"\bstart of(?: the)? day\b", r"\bdaybreak\b", r"\bwake up\b",
        r"\bcrack of dawn\b", r"\brise and shine\b", r"\bfirst light\b",
        r"\bearly hours\b", r"\bcoffee time\b", r"\bmorning time\b"
    ],
    "afternoon": [
        r"\bafternoon[s]?\b", r"\bnoon\b", r"\bmidday\b", r"\bp\.?m\.?\b",
        r"\blunch(?:time)?\b", r"\bsecond (?:place|spot)\b", r"\bmatinee\b",
        r"\bdaytime\b", r"\bpost[- ]lunch\b", r"\bmid[- ]afternoon\b",
        r"\blate afternoon\b", r"\btea time\b", r"\bgolden hour\b",
        r"\bsiesta\b", r"\bduring the day\b"
    ],
    "dinner": [
        r"\bdinner\b", r"\bsupper\b", r"\b(?:rest|din)a[u]?rant[s]?\b",
        r"\bfood\b", r"\b(?:place|somewhere) to eat\b", r"\bbite(?: to eat)?\b",
        r"\bgrub\b", r"\bdining\b", r"\bmeal\b", r"\bstarving\b", r"\bhungry\b",
        r"\bcafe\b", r"\bstreet food\b", r"\bevening meal\b", r"\bchow down\b",
        r"\bgrab dinner\b", r"\bbook a table\b", r"\breservation[s]?\b",
        r"\bfeast\b", r"\bfeed me\b", r"\bdinner time\b",
        r"\bwhere to eat\b", r"\btakeout\b"
    ],
    "hotel": [
        r"\bhotel[s]?\b", r"\bstay(?:ing)?\b", r"\baccommodation[s]?\b",
        r"\blodge(?:s|ing)?\b", r"\bresort[s]?\b", r"\bhostel[s]?\b",
        r"\bairbnb\b", r"\bmotel[s]?\b", r"\binn\b", r"\bbooking[s]?\b",
        r"\breservation[s]?\b", r"\bsleep\b", r"\bguest[- ]house\b", r"\bb&b\b",
        r"\bplace to crash\b", r"\bcheck[- ]in\b", r"\bwhere we sleep\b",
        r"\bsuite[s]?\b", r"\bvilla[s]?\b", r"\bhomestay[s]?\b",
        r"\bcamping\b", r"\bglamping\b", r"\btent\b",
        r"\bbasecamp\b", r"\bwhere to stay\b"
    ],
}

VISITED_KEYWORDS = [
    r"\balready (?:visited|been|seen|went|did|covered|done)\b",
    r"\b(?:i've|i have|we've|we have) (?:been|visited|seen)\b",
    r"\bbeen there\b", r"\bdone that\b", r"\bchecked (?:it|that) off\b",
    r"\bfamiliar with\b", r"\bsaw (?:it|that)\b", r"\bwent last time\b",
    r"\bsaw it already\b", r"\bknocked that out\b", r"\bnot my first time\b",
    r"\bpreviously visited\b", r"\bi know it well\b",
    r"\bwe went there\b", r"\bcrossed that off\b"
]

DISLIKE_KEYWORDS = [
    r"\bdon'?t like\b",
    r"\bnot (?:good|happy|interested|a fan|feeling it|my thing|my vibe)\b",
    r"\b(?:boring|bad|terrible|awful|sucks|hate)\b",
    r"\b(?:too )?(?:touristy|crowded|expensive|loud)\b",
    r"\b(?:skip|pass|avoid|rather not)\b",
    r"\boverrated\b", r"\bhard pass\b", r"\bno thanks\b",
    r"\bdefinitely not\b", r"\bno way\b", r"\btourist trap\b",
    r"\blame\b", r"\bdull\b", r"\bmeh\b", r"\bnah\b", r"\bzero interest\b",
    r"\blooks awful\b", r"\bsounds terrible\b", r"\bwaste of time\b"
]

CHANGE_SLOT_KEYWORDS = [
    r"\b(?:change|replace|swap|switch|modify|update|edit)(?: the| out| it)?\b",
    r"\b(?:different|another|new)\b",
    r"\b(?:suggest|give me) another\b",
    r"\binstead(?: of)?\b",
    r"\balternative(?:s| to)?\b",
    r"\bother than\b", r"\bnot (?:the|this|that)\b",
    r"\bsomething else\b", r"\bwhat else\b",
    r"\b(?:prefer|rather) (?:to )?(?:do|go)\b",
    r"\bscrap (?:that|this)\b", r"\boptions for\b",
    r"\bpivot\b", r"\bmix it up\b", r"\bshuffle\b", r"\bswitch it up\b",
    r"\btrade\b", r"\bsubstitute\b", r"\bfresh idea[s]?\b",
    r"\banything else\b", r"\bwhat else you got\b",
    r"\bgot anything else\b", r"\bredo\b",
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
                if re.search(kw, q):
                    return slot
        return None
    except Exception as e:
        print(f"Error in extract_slot: {e}")
        return None


def extract_correction_type(q):
    try:
        def has_match(kws):
            return any(re.search(kw, q) for kw in kws)

        if has_match(VISITED_KEYWORDS):
            return "visited"

        slot = extract_slot(q)

        # any change keyword locks us into change_slot; the state machine will
        # ask which day/slot if those details are missing
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
            return any(re.search(kw, q) for kw in kws)
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


def fill_slot(slot_name, raw_answer, session):
    try:
        answer = raw_answer.strip().lower()

        if slot_name == "destination":
            val = raw_answer.strip().title()
            session["collected"]["destination"] = val
            session["last_location"] = val
            return True

        if slot_name == "days":
            match = re.search(r'\b(\d+)\b', answer)
            if match:
                num = int(match.group(1))
                session["collected"]["days"] = num
                session["trip_days"] = num
                return True
            for word, num in WORD_TO_NUM.items():
                if re.search(rf'\b{word}\b', answer):
                    session["collected"]["days"] = num
                    session["trip_days"] = num
                    return True
            return False

        if slot_name == "budget":
            amount_match = re.search(r'([\d,]+)', answer)
            if amount_match:
                amount = float(amount_match.group(1).replace(',', ''))
                session["collected"]["budget"] = amount
                session["preferences"]["budget"] = amount
                return True
            if any(w in answer for w in ["budget", "cheap", "low", "economic"]):
                val = "budget"
            elif any(w in answer for w in ["luxury", "high", "premium", "expensive"]):
                val = "luxury"
            else:
                val = "mid-range"
            session["collected"]["budget"] = val
            session["preferences"]["budget"] = val
            return True

        return False
    except Exception as e:
        print(f"Error in fill_slot: {e}")
        return False


def handle_correction_followup(user_input, session):
    try:
        q = user_input.lower()
        correction = session.get("pending_correction") or parse_correction(user_input) or {}

        if correction.get("day") is None:
            day = extract_day_number(q)
            if day is None:
                session["pending_correction"] = correction
                return None, "Which day would you like me to change?"
            correction["day"] = day

        if correction.get("type") in ("change_slot", "visited") and correction.get("slot") is None:
            slot = extract_slot(q)
            if slot is None:
                session["pending_correction"] = correction
                return None, "Which part — morning, afternoon, dinner, or hotel?"
            correction["slot"] = slot

        session["pending_correction"] = None
        return correction, None

    except Exception as e:
        print(f"Error in handle_correction_followup: {e}")
        return session.get("pending_correction"), None
