import re

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
    }
}

# Higher index = higher priority when multiple intents match
INTENT_PRIORITY = ['location', 'restaurant', 'taxicab', 'hotel', 'distance']


def detect_intent(query):
    query_lower = query.lower()

    distance_specific = ['distance', 'how far', 'km', 'miles', 'haversine']
    if any(re.search(r'\b' + re.escape(kw) + r'\b', query_lower) for kw in distance_specific):
        return 'distance'

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
    """Re-evaluates intent from query; keeps current_intent when the new query only adds
    a location modifier (e.g. 'nearby'). Unrecognized queries (None) and greetings reset context."""
    new_intent = detect_intent(query)
    if new_intent is None:
        return None
    if new_intent == 'location' and current_intent not in ('location', None, 'greeting'):
        return current_intent
    return new_intent


def add_intent(name, keywords):
    """Dynamically registers a new intent and appends it to the priority list."""
    key = f"{name}-keywords"
    intents[key] = {"keywords": list(keywords)}
    if name not in INTENT_PRIORITY:
        INTENT_PRIORITY.append(name)


