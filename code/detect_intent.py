import re

# The preprocess_location_query function takes a user's query and removes any budget-related information, day/night counts, trip planning phrases, and common filler words to isolate the core location or destination mentioned in the query, which can then be used for more accurate matching against a dataset of locations.
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
# The reset_session function clears all stored information in the session, including the last location, intent, results, trip duration, and user preferences, effectively resetting the state of the assistant to handle a new conversation or query without any prior context.
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

# The detect_intent function analyzes the user's query to determine the underlying intent by checking for specific keywords and patterns associated with different intents, such as distance queries, itinerary planning, and location-based requests, and returns the identified intent in a structured format for further processing by the assistant.
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

# The calculate_similarity function computes a fuzzy string similarity score between the user's query and a target string, which helps in ranking potential matches based on how closely they resemble the query.
def wrong_intent(intent):
    if not isinstance(intent, dict):
        print("I'm here to help with specific recommendations and places. What would you like to know?")
        return
    for j in intent.values():
        if j == True:
            return True
    print("I'm here to help with specific recommendations and places. What would you like to know?")

# The calculate_similarity function computes a fuzzy string similarity score between the user's query and a target string, which helps in ranking potential matches based on how closely they resemble the query.
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

# The calculate_similarity function computes a fuzzy string similarity score between the user's query and a target string, which helps in ranking potential matches based on how closely they resemble the query.
def add_intent(name, keywords):
 
    key = f"{name}-keywords"
    intents[key] = {"keywords": list(keywords)}
    if name not in INTENT_PRIORITY:
        INTENT_PRIORITY.append(name)