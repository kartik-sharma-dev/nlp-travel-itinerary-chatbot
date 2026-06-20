# NLP Travel Itinerary Chatbot

A full-stack Django web application with a real-time WebSocket chatbot that helps users plan trips across India — find hotels and restaurants, calculate distances, and generate day-by-day itineraries — all through natural language conversation.

---

## Features

- **Itinerary Builder** — generates day-by-day travel plans with sightseeing stops, hotels, and restaurants
- **Hotel & Restaurant Finder** — finds accommodations and dining options near any location, ranked by proximity and relevance
- **Distance Calculator** — calculates the straight-line distance between two places and estimates travel time by car, train, and flight
- **Context-Aware Conversation** — remembers your destination and preferences across the chat session; supports corrections like "change day 2's hotel"
- **Real-Time Chat** — WebSocket-based interface with persistent chat history, multiple sessions, and auto-generated titles
- **User Accounts** — signup, login, and per-user conversation storage

---

## Screenshots

**Itinerary Builder** — generates a day-by-day travel plan with sightseeing stops, hotels, and restaurants:

![Itinerary Builder](screenshots/itinerary_builder.png)

**Distance Calculator** — shows straight-line distance and estimated travel times by car, train, and flight:

![Distance Calculator](screenshots/distance_calculator.png)

---

## How It Works

### NLP Pipeline

Queries go through a multi-stage pipeline inside [`user/nlp_bridge.py`](user/nlp_bridge.py):

1. **Intent detection** ([`detect_intent.py`](user/nlp_engine/detect_intent.py)) — keyword + regex matching classifies the query into one of 14 intents (`build_itinerary`, `hotel`, `restaurant`, `distance_query`, `greeting`, `refine_itinerary`, `confirm`, `restart`, etc.)

2. **Entity extraction** ([`query_utils.py`](user/nlp_engine/query_utils.py)) — spaCy NER + regex pulls out location names, day counts, and budget figures from free-form sentences

3. **Fuzzy + TF-IDF scoring** ([`scoring.py`](user/nlp_engine/scoring.py)) — candidates from the CSV dataset are ranked using a combined score:
   - **70% RapidFuzz** (weighted across place name, state, and country fields)
   - **30% TF-IDF cosine similarity** (character n-gram 3–5)

4. **Geocoding fallback** ([`handlers.py`](user/nlp_engine/handlers.py)) — when a place is not in the dataset, the app queries the Nominatim (OpenStreetMap) API to resolve coordinates, then saves them to the CSV for future sessions

5. **Haversine routing** — vectorized NumPy + SciPy haversine calculates real distances between stops; `build_chain` assembles an optimized visit order using NetworkX graphs

### State Machine

The conversation follows a four-state machine: `greeting → collecting_info → reviewing → finalized`. Missing slots (destination, days) are asked one at a time. Corrections like "swap day 2's morning" trigger a follow-up question flow that blacklists the rejected place/hotel/restaurant and regenerates only the affected day.

### Real-Time Communication

[`consumers.py`](user/consumers.py) is a Django Channels `WebsocketConsumer`. Each browser connection gets its own bot session dictionary. Messages flow:

```
Browser ──WS──▶ ChatConsumer.receive()
                    ├── summary trigger → summary_function()
                    └── nlp_bridge.get_response() → bot reply
                ◀──WS── JSON response
```

---

## Architecture

```
Browser
  │  WebSocket (ws://)
  ▼
consumers.py  ──────────────────────────────────────────────┐
ChatConsumer.receive()                                       │
  │  summary trigger → helper_functions.summary_function()  │
  │                                                          │
  ▼                                                          │
nlp_bridge.get_response()                                    │
  │                                                          │
  ├─ detect_intent.py                                        │
  │    keyword + regex → intent label                        │
  │    four-state machine (greeting / collecting_info /      │
  │                         reviewing / finalized)           │
  │                                                          │
  ├─ query_utils.py                                          │
  │    spaCy NER + regex → location, days, budget            │
  │                                                          │
  └─ handlers.py  (hotel / restaurant / distance /           │
       │            itinerary)                               │
       │                                                     │
       ├─ scoring.py                                         │
       │    score_and_rank()                                  │
       │      70% RapidFuzz (place + state + country)        │
       │      30% TF-IDF cosine similarity (char n-gram 3–5) │
       │    build_chain()                                     │
       │      Haversine distances → NetworkX greedy route    │
       │                                                     │
       └─ Nominatim API  (geocoding fallback)                │
            ↳ result appended to real_landmark_locations.csv │
                                                             │
  ◀──────────────────────────── JSON response ───────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Django 6 + Django Channels 4 |
| WebSocket server | Daphne (ASGI) |
| Async runtime | Gevent + Twisted |
| NLP | spaCy `en_core_web_md` + `en_core_web_sm` |
| Deep learning | PyTorch 2.12 + HuggingFace Transformers |
| Fuzzy matching | RapidFuzz |
| ML / similarity | scikit-learn (TF-IDF + cosine similarity) |
| Math / routing | NumPy, SciPy, NetworkX |
| Data | pandas |
| Geocoding | Nominatim (OpenStreetMap) via `requests` |
| Database | SQLite (default) |
| Auth | Django custom user model (`AbstractUser`) |

---

## Project Structure

```
nlp-travel-itinerary-chatbot/
├── traveler/                   # Django project config (settings, urls, asgi)
├── user/
│   ├── nlp_engine/             # NLP engine
│   │   ├── detect_intent.py    # Intent classifier + session state
│   │   ├── handlers.py         # Hotel, restaurant, distance, itinerary handlers
│   │   ├── load_location_data.py
│   │   ├── preprocess.py       # spaCy lemmatizer + stopword removal
│   │   ├── query_utils.py      # Location / days / budget extractors
│   │   └── scoring.py          # Fuzzy + TF-IDF ranking + Haversine chain
│   ├── Data/
│   │   ├── real_landmark_locations.csv   # Places, lat/lon, ratings
│   │   └── large_nlp_recommendations.csv
│   ├── consumers.py            # WebSocket consumer
│   ├── nlp_bridge.py           # Main brain — routes intents to handlers
│   ├── models.py               # Custom_user, Chat_Title, Conversation
│   ├── views.py                # Auth views (signup, login, logout)
│   ├── helper_functions.py     # Stopword cleaner, chat summary
│   ├── authentication.py
│   ├── routing.py              # WebSocket URL routing
│   └── templates/user/
│       └── chatbot.html        # Chat UI
├── place_distances_edge_list.csv
├── manage.py
└── requirements.txt
```

---

## Setup

**Prerequisites:** Python 3.12, pip

```bash
# 1. Clone and enter the project
git clone https://github.com/<your-username>/nlp-travel-itinerary-chatbot.git
cd nlp-travel-itinerary-chatbot

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run migrations
python manage.py migrate

# 5. Start the server
python manage.py runserver
```

Open `http://localhost:8000` in your browser.

> **Note:** `en_core_web_md` and `en_core_web_sm` are bundled in `requirements.txt` via spaCy GitHub release URLs — no separate `spacy download` step needed.

---

## Example Conversations

```
You: Plan a 3-day trip to Jaipur with a budget of 15000
Bot: 🗺️ 3-Day Itinerary for Jaipur
     📅 DAY 1
       ✨ Stop 1: Amber Fort
       ✨ Stop 2: Nahargarh Fort
       🏨 Hotel: Dera Mandawa
       🍽️  Restaurant: Suvarna Mahal
     ...

You: Change day 2's hotel
Bot: Which part — morning, afternoon, dinner, or hotel?

You: hotel
Bot: 🗺️ Updated itinerary for Jaipur...
```

```
You: Distance between Delhi and Agra
Bot: 📍 Distance & Route Information
     From: Delhi, Delhi
     To:   Agra, Uttar Pradesh
     📏 Distance: 207.4 km
     🚗 By Car:   3h 27m
     🚂 By Train: 2h 4m
     ✈️  By Flight: 0h 15m
```

---

## Data

The app uses two CSVs under [`user/Data/`](user/Data/):

- `real_landmark_locations.csv` — landmarks, hotels, and restaurants with columns: `location`, `landmark`, `state`, `country`, `lon_location`, `lat_location`, `type`, `ratings`, `total_reviews`
- `large_nlp_recommendations.csv` — supplemental NLP recommendation data

When a queried place is not in the dataset, the app geocodes it via Nominatim and appends the result to the CSV automatically.

---

## Known Limitations & What I'd Do Differently

**Rule-based intent detection is brittle.** The classifier is keyword + regex, so phrasing it doesn't anticipate breaks it ("can you suggest a place to stay" misses the hotel intent). A small fine-tuned text classifier — or even a few-shot prompt to a smaller LLM — would generalize much better without much extra complexity.

**Synchronous NLP blocks the event loop.** The spaCy + TF-IDF pipeline runs synchronously inside the WebSocket consumer. Under concurrent connections this stalls Daphne. The fix is either `asyncio.to_thread` to offload the CPU work, or a dedicated task queue (Celery + Redis).

**Session state lives only in memory.** Bot sessions are plain Python dicts scoped to each consumer instance. A server restart or Daphne crash silently loses every in-progress conversation. Persisting sessions to Redis (which Channels already supports for the channel layer) would fix this.

**CSV as a mutable data store is fragile.** The geocoding fallback appends new rows directly to the CSV at runtime. Concurrent writes can corrupt it and there's no schema enforcement. A proper DB table (even still SQLite) would be safer and faster to query.

**Greedy nearest-neighbor isn't a real route optimizer.** `build_chain` picks the closest unvisited stop at each step. For 3–4 stops this is fine, but it won't find the globally best order. A small TSP solver (OR-Tools, or even brute-force up to ~8 stops) would produce noticeably better day plans.

**The tech-stack lists PyTorch + Transformers but they're not used for inference.** They're installed (likely a dependency of spaCy's transformer pipeline), but the actual NLP is TF-IDF + RapidFuzz + spaCy's statistical model. Worth clarifying, or worth actually wiring in a small transformer for entity extraction where spaCy's NER misses Indian place names.

---

## Dependencies

- [Django](https://www.djangoproject.com/) + [Channels](https://channels.readthedocs.io/) — web + WebSocket framework
- [Daphne](https://github.com/django/daphne) — ASGI server for WebSocket support
- [spaCy](https://spacy.io/) — NLP preprocessing and NER
- [PyTorch](https://pytorch.org/) + [HuggingFace Transformers](https://huggingface.co/docs/transformers/) — deep learning backbone
- [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) — fast fuzzy string matching
- [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorizer + cosine similarity
- [SciPy](https://scipy.org/) — spatial math and distance calculations
- [NetworkX](https://networkx.org/) — graph-based route chaining
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling and vectorized math
- [Gevent](https://www.gevent.org/) + [Twisted](https://twisted.org/) — async concurrency
