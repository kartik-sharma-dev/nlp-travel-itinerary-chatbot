import re
from preprocess import nlp

INDIA_LOCATIONS = {
    # Delhi
    'delhi': 'Delhi',
    'new delhi': 'Delhi',
    # Uttar Pradesh cities → map to actual landmarks/state in dataset
    'agra': 'Taj Mahal',
    'varanasi': 'Kashi Vishwanath Temple',
    'lucknow': 'Bara Imambara',
    'mathura': 'Prem Mandir',
    'allahabad': 'Triveni Sangam',
    'prayagraj': 'Triveni Sangam',
    'ayodhya': 'Ram Janmabhoomi',
    # Maharashtra cities
    'mumbai': 'Maharashtra',
    'bombay': 'Maharashtra',
    'pune': 'Shaniwar Wada',
    'nashik': 'Maharashtra',
    'aurangabad': 'Ajanta Caves',
    # Rajasthan cities
    'jaipur': 'Rajasthan',
    'jodhpur': 'Mehrangarh Fort',
    'udaipur': 'City Palace Udaipur',
    'jaisalmer': 'Jaisalmer Fort',
    'pushkar': 'Pushkar Lake',
    'ajmer': 'Rajasthan',
    'bikaner': 'Rajasthan',
    # Goa
    'goa': 'Goa',
    'panaji': 'Goa',
    'panjim': 'Goa',
    # Kerala cities
    'kerala': 'Kerala',
    'munnar': 'Munnar Tea Gardens',
    'alleppey': 'Alleppey Backwaters',
    'alappuzha': 'Alleppey Backwaters',
    'kochi': 'Kochi Chinese Fishing Nets',
    'cochin': 'Kochi Chinese Fishing Nets',
    'thiruvananthapuram': 'Padmanabhaswamy Temple',
    'trivandrum': 'Padmanabhaswamy Temple',
    # Karnataka cities
    'bangalore': 'Karnataka',
    'bengaluru': 'Karnataka',
    'mysore': 'Mysore Palace',
    'mysuru': 'Mysore Palace',
    'hampi': 'Hampi Ruins',
    'coorg': 'Coorg Coffee Plantations',
    'kodagu': 'Coorg Coffee Plantations',
    'chikmagalur': 'Chikmagalur',
    # Tamil Nadu cities
    'chennai': 'Tamil Nadu',
    'madras': 'Tamil Nadu',
    'madurai': 'Meenakshi Amman Temple',
    'ooty': 'Ooty Botanical Gardens',
    'kodaikanal': 'Kodaikanal Lake',
    'kanyakumari': 'Kanyakumari Beach',
    'mahabalipuram': 'Mahabalipuram Shore Temple',
    # Gujarat cities
    'gujarat': 'Gujarat',
    'ahmedabad': 'Sabarmati Ashram',
    'vadodara': 'Laxmi Vilas Palace',
    'baroda': 'Laxmi Vilas Palace',
    'kutch': 'Rann of Kutch',
    'rann of kutch': 'Rann of Kutch',
    # West Bengal cities
    'kolkata': 'West Bengal',
    'calcutta': 'West Bengal',
    'darjeeling': 'Darjeeling Himalayan Railway',
    'siliguri': 'West Bengal',
    # Punjab cities
    'amritsar': 'Golden Temple',
    'chandigarh': 'Rock Garden Chandigarh',
    'ludhiana': 'Punjab',
    # Madhya Pradesh
    'bhopal': 'Upper Lake Bhopal',
    'khajuraho': 'Khajuraho Group of Monuments',
    'ujjain': 'Mahakaleshwar Jyotirlinga',
    'gwalior': 'Gwalior Fort',
    'jabalpur': 'Bhedaghat Marble Rocks',
    # Himachal Pradesh
    'shimla': 'Shimla',
    'manali': 'Manali',
    'dharamsala': 'Dharamsala',
    'mcleod ganj': 'Dharamsala',
    'kasol': 'Kasol',
    'kullu': 'Kullu Valley',
    'spiti': 'Spiti Valley',
    # Uttarakhand
    'mussoorie': 'Mussoorie',
    'dehradun': 'Dehradun',
    'haridwar': 'Haridwar',
    'rishikesh': 'Rishikesh',
    'nainital': 'Nainital',
    # Jammu & Kashmir / Ladakh
    'srinagar': 'Dal Lake Srinagar',
    'leh': 'Leh',
    'ladakh': 'Leh',
    # Northeast India
    'shillong': 'Shillong',
    'gangtok': 'Gangtok',
    # Andhra Pradesh / Telangana
    'hyderabad': 'Hyderabad',
    'visakhapatnam': 'Visakhapatnam',
    'vizag': 'Visakhapatnam',
    # Famous landmarks (direct)
    'taj mahal': 'Taj Mahal',
    'taj': 'Taj Mahal',
    'red fort': 'Red Fort',
    'india gate': 'India Gate',
    'qutub minar': 'Qutub Minar',
    'gateway of india': 'Gateway of India',
    'golden temple': 'Golden Temple',
    'hawa mahal': 'Hawa Mahal',
    'amber palace': 'Amber Palace',
    'mysore palace': 'Mysore Palace',
    'ajanta caves': 'Ajanta Caves',
    'ellora caves': 'Ellora Caves',
}

def extract_location(query):
    """Used for standalone queries (e.g., 'hotels near Mumbai')"""
    try:
        q = query.lower()

        # 1. Quick check against standard cities
        for loc_key, loc_value in INDIA_LOCATIONS.items():
            if re.search(rf'\b{re.escape(loc_key)}\b', q):
                return loc_value

        # 2. Safely strip leading conversational filler (without destroying real names)
        fillers = [
            r'\bi am near\b', r'\bi am at\b', r'\bcan you tell me\b', 
            r'\bsome nearby\b', r'\bwhere i can stay\b', r'\bclose to\b',
            r'\bhotels near\b', r'\bhotel near\b', r'\brestaurants in\b',
            r'\bplaces to visit in\b', r'\bshow me\b', r'\bfind me\b',
            r'\bnearby\b', r'\bnear\b', r'\baround\b'
        ]
        
        for filler in fillers:
            q = re.sub(filler, ' ', q)
            
        # Strip out money and numbers so they aren't treated as places
        q = re.sub(r'(?:rs\.?|inr|₹|rupees?)', '', q)
        q = re.sub(r'\b\d+(?:,\d{3})*\b', '', q)
        
        return re.sub(r'\s+', ' ', q).strip()
    except Exception as e:
        print(f"Error in extract_location: {e}")
        return query


def extract_entities_bio(query):
    """Uses spaCy NER to find places."""
    try:
        if nlp is None: return []
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
    """The deep extractor for complex trip planning sentences."""
    try:
        q = query.lower()

        # 1. Try NER first — most reliable when spaCy recognises the city
        doc = nlp(query)
        gpe_ents = [ent.text for ent in doc.ents if ent.label_ in ('GPE', 'LOC', 'FAC')]
        if gpe_ents:
            return gpe_ents[0].lower().strip()

        # 2. Check the manual dictionary fallback
        for loc_key, loc_value in INDIA_LOCATIONS.items():
            if re.search(rf'\b{re.escape(loc_key)}\b', q):
                return loc_value

        # 3. Strip budget patterns first so numbers don't confuse later steps
        q = re.sub(r'budget\s*(?:of|is|:)?\s*(?:rs\.?|inr|₹|rupees?)?\s*[\d,]+', '', q)
        q = re.sub(r'(?:rs\.?|inr|₹)\s*[\d,]+', '', q)
        q = re.sub(r'\b(?:under|within|less\s+than)\s+(?:rs\.?|inr|₹|rupees?)?\s*[\d,]+', '', q)

        # 4. Strip day/night counts
        word_nums = 'one|two|three|four|five|six|seven|eight|nine|ten'
        q = re.sub(rf'\b(?:{word_nums}|\d+)\s*(?:day|days|night|nights)\b', '', q)

        # 5. Strip trip planning verbs
        trip_phrases = [
            'give me a travel plan for', 'plan a trip to', 'plan my trip to',
            'plan a trip for', 'travel plan for', 'itinerary for',
            'i want to visit', 'i want to go to', 'trip to', 'travel to',
            'plan my trip', 'plan a trip',
            'plan a holiday to', 'plan my holiday to', 'plan a holiday',
            'plan my holiday', 'plan a vacation to', 'plan my vacation',
            'plan a vacation', 'have a holiday',
        ]
        for phrase in sorted(trip_phrases, key=len, reverse=True):
            q = re.sub(rf'\b{re.escape(phrase)}\b', ' ', q)

        # Clean up stray artifacts
        q = re.sub(r'\b(?:rs\.?|inr|rupees?)\b', ' ', q)
        q = re.sub(r'\b\d+\b', ' ', q)

        result = re.sub(r'\s+', ' ', q).strip()

        # Strip leftover leading prepositions after phrase removal ("for shimla" → "shimla")
        result = re.sub(r'^(?:for|to|in|at|near|around|from|on|about)\s+', '', result).strip()

        # Discard results that are only filler / stop words with no actual place name
        _FILLER = {
            'i', 'want', 'to', 'go', 'am', 'the', 'a', 'an', 'at', 'in', 'for',
            'me', 'my', 'trip', 'plan', 'visit', 'need', 'please', 'help',
            'we', 'us', 'change', 'destination', 'somewhere', 'different',
            'some', 'is', 'are', 'was', 'be', 'do', 'did', 'does', 'its',
            'holiday', 'vacation', 'have', 'thinking', 'of', 'going', 'travel',
            'you', 'can', 'so', 'could', 'would', 'should', 'hey', 'ok',
            'okay', 'get', 'build', 'make', 'show', 'create', 'give', 'just',
            'your', 'tell', 'what', 'where', 'when', 'how', 'with', 'and',
        }
        words = result.lower().split()
        if not words or all(w in _FILLER for w in words):
            return ''

        return result
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