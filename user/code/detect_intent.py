import re


session = {
    "last_location" : None,
    "last_intent"   : None,
    "last_results"  : None,
    "trip_days"     : None,
    
    "preferences"   : {
        "veg_only"  : False,
        "budget"    : None,
    }
}

def reset_session():
    session["last_location"] = None
    session["last_intent"]   = None
    session["last_results"]  = None
    session["trip_days"]     = None
    session["preferences"]   = {
        "veg_only" : False,
        "budget"   : None,
    }

intents = {
    "greeting-keywords": {
        "keywords": [
            'hi', 'hello', 'hey', 'hiya', 'yo', 'sup', 'greetings', 'howdy',
            'namaste', 'namaskar', 'salaam', 'good morning', 'good afternoon',
            'good evening', 'good night', 'how are you', "what's up",
            'hy', 'hii', 'helo', 'heya', 'hai'
        ]
    },
    "hotel-keywords": {
        "keywords": ['hotel', 'room', 'stay', 'accommodation', 'lodge', 'resort', 'book']
    },
    "location-keywords": {
        "keywords": ['nearby', 'near', 'close to', 'around', 'some nearby', 'where i can stay', 'i can stay', 'can stay']
    },
    "distance-query-keywords": {
        "keywords": ['distance', 'how far', 'km', 'miles', 'haversine']
    },
    "restaurant-keywords": {
        "keywords": ['restaurant', 'food', 'dining', 'eatery', 'cafe', 'diner', 'bistro', 'eat']
    },
    "taxicab-keywords": {
        "keywords": ['taxi', 'cab', 'ride', 'transport', 'uber', 'lyft', 'taxicab', 'chauffeur']
    },
    "itinerary-keywords": {
        "keywords": ['itinerary', 'trip plan', 'travel plan', 'plan my trip', 'plan a trip', 'day tour', 'day trip']
    }
}

INTENT_PRIORITY = ['location', 'restaurant', 'taxicab', 'hotel', 'distance', 'itinerary']


def detect_intent(query):
    query_lower = query.lower()

    distance_specific = ['distance', 'how far', 'km', 'miles', 'haversine']
    if any(re.search(r'\b' + re.escape(kw) + r'\b', query_lower) for kw in distance_specific):
        return 'distance'

    _word_nums = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    _has_days = re.search(r'\b\d+\s*(?:day|days|night|nights)\b', query_lower) or \
                any(re.search(rf'\b{w}\s+(?:day|days|night|nights)\b', query_lower) for w in _word_nums)
    _itin_kws = ['itinerary', 'trip plan', 'travel plan', 'plan my trip', 'plan a trip']
    if _has_days or any(kw in query_lower for kw in _itin_kws):
        return 'itinerary'

    matched = []
    for intent_key, details in intents.items():
        intent_name = intent_key.split('-')[0]
        for keyword in details['keywords']:
            if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                matched.append(intent_name)
                break

    if not matched:
        return None

    for intent in reversed(INTENT_PRIORITY):
        if intent in matched:
            if intent == 'location':
                return 'location'
            return {intent: True}

    return matched[0]


def wrong_intent(intent):
    if not isinstance(intent, dict):
        print("I'm here to help with specific recommendations and places. What would you like to know?")
        return
    for j in intent.values():
        if j == True:
            return True
    print("I'm here to help with specific recommendations and places. What would you like to know?")


def update_intent(current_intent, query):

    new_intent = detect_intent(query)

    if new_intent is None:
        return None

    if new_intent == 'location' and current_intent not in ('location', None, 'greeting'):
        return current_intent

    # update session
    session["last_intent"] = new_intent

    # reset session on greeting
    if new_intent == 'greeting':
        reset_session()

    return new_intent


def add_intent(name, keywords):
 
    key = f"{name}-keywords"
    intents[key] = {"keywords": list(keywords)}
    if name not in INTENT_PRIORITY:
        INTENT_PRIORITY.append(name)