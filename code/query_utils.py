import re
from preprocess import nlp

# The extract_location function takes a user query and removes common filler words and phrases to isolate the core location information, making it easier for subsequent processing steps to identify the intended location.
def extract_location(query):
    fillers = sorted([
        'i am near', 'i am at', 'i am', 'can you tell me', 'can you tell',
        'some nearby hotels', 'some nearby hotel', 'some nearby', 'nearby hotels',
        'nearby hotel', 'where i can stay', 'i can stay', 'can stay', 'close to',
        'i need', 'a room', 'around', 'nearby', 'near', 'people', 'persons', 'guests',
        'hotels', 'hotel', 'some', 'where', 'stay', 'tell', 'for',
        'hy', 'hi', 'hey', 'hello', 'me', 'you', 'i', 'can'
    ], key=len, reverse=True)

    query = query.lower()
    for filler in fillers:
        query = re.sub(rf'\b{re.escape(filler)}\b', ' ', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query

# The calculate_similarity function computes a fuzzy string similarity score between the user's query and a target string, which helps in ranking potential matches based on how closely they resemble the query.
def extract_entities_bio(query):
    doc = nlp(query)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ('GPE', 'LOC', 'FAC'):
            entities.append(ent.text.lower().strip())
    return entities

# The extract_itinerary_location function is designed to parse user queries related to travel itineraries, extracting the intended destination while removing extraneous information such as budget, duration, and common trip-related phrases, thus isolating the core location for itinerary planning.
def extract_itinerary_location(query):
    # Try NER first — most reliable when spaCy recognises the city
    doc = nlp(query)
    gpe_ents = [ent.text for ent in doc.ents if ent.label_ in ('GPE', 'LOC', 'FAC')]
    if gpe_ents:
        return gpe_ents[0].lower().strip()

    q = query.lower()

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
        'plan a', 'plan my', 'trip to', 'travel to', 'travel plan',
        'trip plan', 'plan', 'trip', 'travel', 'itinerary', 'visit', 'explore', 'tour',
        'create', 'generate', 'make', 'build', 'suggest', 'show',
    ], key=len, reverse=True)
    for phrase in trip_phrases:
        q = re.sub(rf'\b{re.escape(phrase)}\b', ' ', q)

    # strip leftover prepositions / filler words
    fillers = ['i am in', 'i am at', 'i am', 'in', 'to', 'for', 'at', 'a', 'an',
               'the', 'with', 'give', 'me', 'my', 'some', 'please', 'want', 'am', 'is', 'are']
    for w in sorted(fillers, key=len, reverse=True):
        q = re.sub(rf'\b{re.escape(w)}\b', ' ', q)

    # strip any stray currency words or lone numbers left after budget stripping
    q = re.sub(r'\b(?:rs\.?|inr|rupees?)\b', ' ', q)
    q = re.sub(r'\b\d+\b', ' ', q)

    return re.sub(r'\s+', ' ', q).strip()

# The extract_days function identifies and extracts the number of days or nights mentioned in a user's query, which is crucial for planning itineraries and providing relevant recommendations based on the duration of the trip.
def extract_days(query):
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

# The extract_budget function parses the user's query to identify any mentioned budget constraints, extracting numerical values associated with currency indicators or budget-related phrases, which can then be used to filter recommendations based on the user's financial preferences.
def extract_budget(query):
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