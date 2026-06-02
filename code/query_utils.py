import re
import spacy
from preprocess import number_words

nlp = spacy.load('en_core_web_md')


def extract_person_count(query):
    for word, digit in number_words.items():
        query = re.sub(rf'\b{word}\b', digit, query.lower())
    match = re.search(r'\b(\d+)\s+(people|persons?|guests?|adults?|members?)\b', query)
    if match:
        return int(match.group(1))
    return None


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


def extract_entities_bio(query):
    doc = nlp(query)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ('GPE', 'LOC', 'FAC'):
            entities.append(ent.text.lower().strip())
    return entities


