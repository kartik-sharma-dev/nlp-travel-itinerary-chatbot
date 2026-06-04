import spacy

number_words = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
    'nine': '9', 'ten': '10'
}

try:
    nlp = spacy.load('en_core_web_md')
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    nlp = None

def preprocess(text):
    try:
        if not isinstance(text, str):
            text = str(text)
        for word, num in number_words.items():
            text = text.replace(word, num)
        if nlp is None:
            raise RuntimeError("spaCy model is not loaded.")
        doc = nlp(text)
        tokens = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            if token.like_num:
                tokens.append(token.text)
            elif not token.is_stop:
                tokens.append(token.text)
        return " ".join(tokens)
    except Exception as e:
        print(f"Error in preprocess: {e}")
        return text