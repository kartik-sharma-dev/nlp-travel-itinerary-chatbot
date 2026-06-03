
import spacy



# The preprocess function takes a text input, converts number words to digits, removes stop words and punctuation, and returns a cleaned version of the text that is easier for the model to process and analyze.
number_words = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
    'nine': '9', 'ten': '10'
}


nlp = spacy.load('en_core_web_md')

def preprocess(text):
    if not isinstance(text, str):
        text = str(text)
    for word, num in number_words.items():
        text = text.replace(word, num)
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
