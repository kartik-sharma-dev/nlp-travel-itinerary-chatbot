import re
from preprocess import nlp

INDIA_LOCATIONS = {
    'delhi': 'delhi',
    'new delhi': 'delhi',
    'jaipur': 'jaipur',
    'taj mahal': 'taj mahal',
    'taj': 'taj mahal',
    'red fort': 'red fort',
    'agra': 'agra',
    'mumbai': 'mumbai',
    'bangalore': 'bangalore',
    'kolkata': 'kolkata',
    'chennai': 'chennai',
    'hyderabad': 'hyderabad',
    'pune': 'pune',
    'lucknow': 'lucknow',
    'varanasi': 'varanasi',
}

GENERIC_LOCATION_NOISE = {
    'restaurant', 'restaurants', 'restaraunt', 'restaraunts', 'food', 'eat',
    'eating', 'dining', 'cafe', 'cafes', 'diner', 'diners', 'bistro', 'place',
    'places', 'good place', 'good places', 'nearby', 'near', 'some', 'find',
    'show', 'suggest', 'me'
}

def extract_location(query):
    try:
        fillers = sorted([
            'i am near', 'i am at', 'i am', 'can you tell me', 'can you tell',
            'some nearby hotels', 'some nearby hotel', 'some nearby', 'nearby hotels',
            'nearby hotel', 'where i can stay', 'i can stay', 'can stay', 'close to',
            'i need', 'a room', 'around', 'nearby', 'near', 'people', 'persons', 'guests',
            'hotels', 'hotel', 'some', 'where', 'stay', 'tell', 'for', 'find', 'to',
            'good places', 'good place', 'places', 'place', 'eat', 'eating', 'food',
            'restaurants', 'restaurant', 'dining', 'cafe', 'diner', 'bistro',
            'hy', 'hi', 'hey', 'hello', 'me', 'you', 'i', 'can'
        ], key=len, reverse=True)
        query = query.lower()

        for loc_key, loc_value in INDIA_LOCATIONS.items():
            if re.search(rf'\b{re.escape(loc_key)}\b', query):
                return loc_value

        for filler in fillers:
            query = re.sub(rf'\b{re.escape(filler)}\b', ' ', query)
        query = re.sub(r'(?:rs\.?|inr|₹|rupees?)', '', query)
        query = re.sub(r'\b\d+(?:,\d{3})*\b', '', query)
        query = re.sub(r'\b(?:place|places|eat|eating|food|restaurant|restaurants|restaraunt|restaraunts|dining|cafe|diner|bistro)\b', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
        return query
    except Exception as e:
        print(f"Error in extract_location: {e}")
        return query

def extract_entities_bio(query):
    try:
        doc = nlp(query)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ('GPE', 'LOC', 'FAC'):
                entities.append(ent.text.lower().strip())
        return entities
    except Exception as e:
        print(f"Error in extract_entities_bio: {e}")
        return []

def extract_itinerary_location(query):
    try:
        # Try NER first — most reliable when spaCy recognises the city
        doc = nlp(query)
        gpe_ents = [ent.text for ent in doc.ents if ent.label_ in ('GPE', 'LOC', 'FAC')]
        if gpe_ents:
            return gpe_ents[0].lower().strip()

        q = query.lower()

        for loc_key, loc_value in INDIA_LOCATIONS.items():
            if re.search(rf'\b{re.escape(loc_key)}\b', q):
                return loc_value

        if any(re.search(rf'\b{re.escape(term)}\b', q) for term in GENERIC_LOCATION_NOISE):
            return ''

        # strip budget patterns first so numbers don't confuse later steps
        q = re.sub(r'budget\s*(?:of|is|:)?\s*(?:rs\.?|inr|₹|rupees?)?\s*[\d,]+', '', q)
        q = re.sub(r'(?:rs\.?|inr|₹)\s*[\d,]+', '', q)
        q = re.sub(r'\b(?:under|within|less\s+than)\s+(?:rs\.?|inr|₹|rupees?)?\s*[\d,]+', '', q)

        # strip day/night counts ("3 day", "3 days", "two nights" …)
        word_nums = 'one|two|three|four|five|six|seven|eight|nine|ten'
        q = re.sub(rf'\b(?:{word_nums}|\d+)\s*(?:day|days|night|nights)\b', '', q)

        trip_phrases = sorted([
            'give me a travel plan for', 'plan a trip to', 'plan my trip to',
            'plan a trip for', 'plan my trip for', 'travel plan for', 'trip plan for',
            'itinerary for', 'schedule for', 'i want to visit', 'i want to go to',
            'i want to spend', 'spend', 'plan a', 'plan my', 'trip to', 'travel to',
            'travel plan', 'trip plan', 'plan', 'trip', 'travel', 'itinerary',
            'visit', 'explore', 'tour', 'create', 'generate', 'make', 'build',
            'suggest', 'show', 'luxury', 'budget', 'mid-range', 'midrange', 'expensive',
            'cheap', 'night', 'nights', 'stay', 'hotel', 'restaurant', 'eat',
        ], key=len, reverse=True)
        for phrase in trip_phrases:
            q = re.sub(rf'\b{re.escape(phrase)}\b', ' ', q)

        # strip leftover prepositions / filler words
        fillers = ['i am in', 'i am at', 'i am', 'in', 'to', 'for', 'at', 'a', 'an',
                   'the', 'with', 'give', 'me', 'my', 'some', 'please', 'want', 'am',
                   'is', 'are', 'spend', 'spending', 'night', 'nights', 'budget',
                   'luxury', 'mid-range', 'midrange', 'cheap', 'expensive', 'restaurant',
                   'restaurants', 'hotel', 'hotels', 'eat', 'stay']
        for w in sorted(fillers, key=len, reverse=True):
            q = re.sub(rf'\b{re.escape(w)}\b', ' ', q)

        # strip any stray currency words or lone numbers left after budget stripping
        q = re.sub(r'\b(?:rs\.?|inr|rupees?)\b', ' ', q)
        q = re.sub(r'\b\d+\b', ' ', q)

        return re.sub(r'\s+', ' ', q).strip()
    except Exception as e:
        print(f"Error in extract_itinerary_location: {e}")
        return ""

def extract_days(query):
    try:
        query_lower = query.lower()
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        m = re.search(r'\b(\d+)\s*(?:day|days|night|nights)\b', query_lower)
        if m:
            return int(m.group(1))
        for word, num in word_to_num.items():
            if re.search(rf'\b{word}\s+(?:day|days|night|nights)\b', query_lower):
                return num
        return None
    except Exception as e:
        print(f"Error in extract_days: {e}")
        return None

def extract_budget(query):
    try:
        query_lower = query.lower()
        patterns = [
            r'budget\s*(?:of|is|:)?\s*(?:rs\.?|inr|₹|rupees?)?\s*([\d,]+)',
            r'(?:rs\.?|inr|₹)\s*([\d,]+)',
            r'([\d,]+)\s*(?:rs\.?|inr|rupees?)',
            r'under\s+(?:rs\.?|inr|₹)?\s*([\d,]+)',
            r'less\s+than\s+(?:rs\.?|inr|₹)?\s*([\d,]+)',
            r'within\s+(?:rs\.?|inr|₹)?\s*([\d,]+)',
        ]
        for pattern in patterns:
            m = re.search(pattern, query_lower)
            if m:
                return float(m.group(1).replace(',', ''))
        return None
    except Exception as e:
        print(f"Error in extract_budget: {e}")
        return None