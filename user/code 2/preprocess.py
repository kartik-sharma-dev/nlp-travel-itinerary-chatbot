import spacy
import re

number_words = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
    'nine': '9', 'ten': '10'
}

# Compile patterns ONCE, not on every row
number_patterns = [
    (re.compile(rf'\b{word}\b'), num)
    for word, num in number_words.items()
]

try:
    nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])  # ✅ disable unused components
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    nlp = None

def _replace_number_words(text: str) -> str:
    text = text.lower() if isinstance(text, str) else str(text).lower()
    for pattern, num in number_patterns:
        text = pattern.sub(num, text)
    return text

def preprocess(text: str) -> str:
    return preprocess_batch([text])[0]

def preprocess_batch(texts, batch_size=512):
    """Process a list/Series of texts efficiently using nlp.pipe()"""
    if nlp is None:
        raise RuntimeError("spaCy model is not loaded.")
    
    cleaned = [_replace_number_words(t) for t in texts]
    
    results = []
    for doc in nlp.pipe(cleaned, batch_size=batch_size):  # ✅ batched processing
        tokens = [
            token.text
            for token in doc
            if not token.is_stop and not token.is_punct
        ]
        results.append(" ".join(tokens))
    
    return results