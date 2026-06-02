# Functions have been split into focused modules:
#   preprocess.py        - text preprocessing, number_words
#   load_location_data.py - CSV loading, TF-IDF vectorizer setup
#   detect_intent.py     - intent detection
#   query_utils.py       - query parsing, entity extraction, greeting detection
#   scoring.py           - fuzzy/TF-IDF scoring, haversine, chain building
#   handlers.py          - hotel, distance, and location query handlers
# See main.py for the entry point.
