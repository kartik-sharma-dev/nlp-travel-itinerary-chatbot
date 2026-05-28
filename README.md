# GeoFuzzy — Fuzzy Location Finder

A CLI tool that takes a freeform place-name query and returns the best-matching location from a dataset, along with distances to nearby places in the same state/province.

## How it works

1. **NLP preprocessing** — spaCy lemmatizes and removes stopwords from both the query and location names.
2. **Fuzzy matching** — RapidFuzz scores the query against place name, state, and country fields (weighted 70 / 20 / 10).
3. **TF-IDF cosine similarity** — a character n-gram (3–5) TF-IDF matrix re-ranks candidates.
4. **Final score** — `fuzzy × 0.7 + tfidf × 0.3` determines the best match.
5. **Haversine distances** — all places in the matched state are sorted by distance from the best match.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```bash
python loc.py
# Enter your question: Bombay
```

The tool prints the best match and a ranked table of nearby places with distances in km.

## Data

`location.csv` must contain columns: `PlaceName`, `State/Province`, `Country`, `Latitude`, `Longitude`.

## Dependencies

- [spaCy](https://spacy.io/) — NLP preprocessing
- [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) — fuzzy string matching
- [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorizer + cosine similarity
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — data handling
# geofuzzy-location-finder-
