"""
Extended Q&A test suite — uses real dataset landmarks, varied question styles.
Run: python test_bot_extended.py
"""
import sys, os
USER_DIR = os.path.join(os.path.dirname(__file__), "user")
sys.path.insert(0, USER_DIR)

from nlp_bridge import get_response, _new_session

R = "\033[0m"; G = "\033[92m"; RED = "\033[91m"
CY = "\033[96m"; B = "\033[1m"; DIM = "\033[2m"

passed = 0; failed = 0

def new_conv(): return _new_session()

def run(desc, session, q, must_contain=None, must_not=None, label=""):
    global passed, failed
    resp = get_response(q, session)
    tag = f" [{label}]" if label else ""
    print(f"\n{CY}Q{tag}: {q}{R}")
    print(f"A: {resp[:300]}{'...' if len(resp)>300 else ''}")
    ok = True; reasons = []
    if not resp.strip(): ok=False; reasons.append("empty response")
    for kw in (must_contain or []):
        if kw.lower() not in resp.lower(): ok=False; reasons.append(f"missing '{kw}'")
    for kw in (must_not or []):
        if kw.lower() in resp.lower(): ok=False; reasons.append(f"has '{kw}'")
    status = f"{G}PASS{R}" if ok else f"{RED}FAIL{R}"
    rsn = f"  {DIM}({', '.join(reasons)}){R}" if reasons else ""
    print(f"  {B}{status}{R}  {desc}{rsn}")
    if ok: passed+=1
    else:  failed+=1
    return resp

print(f"\n{'='*65}")
print(f"  {B}EXTENDED BOT TEST SUITE — Dataset-aware questions{R}")
print(f"{'='*65}")

# ─── GROUP A: DIFFERENT GREETING STYLES ──────────────────────
print(f"\n{B}─── A: Greeting variations ───{R}")
for phrase in ["hiya", "howdy", "yo", "good morning", "wassup"]:
    s = new_conv()
    run(f"'{phrase}' triggers greeting", s, phrase, must_not=["You said", "couldn't"])

# ─── GROUP B: HOTEL QUERIES — REAL DATASET CITIES ────────────
print(f"\n{B}─── B: Hotel queries (dataset cities) ───{R}")
for city, state_hint in [
    ("Delhi", "Delhi"), ("Goa", "Goa"), ("Kerala", "Kerala"),
    ("Rajasthan", "Rajasthan"), ("Maharashtra", "Maharashtra")
]:
    s = new_conv()
    run(f"Hotels in {city} → India results", s, f"hotels in {city}",
        must_contain=["Hotel", "Rating"], must_not=["You said"])

# ─── GROUP C: RESTAURANT QUERIES — DIFFERENT PHRASINGS ───────
print(f"\n{B}─── C: Restaurant query phrasings ───{R}")
queries = [
    ("where to eat in Goa", "Goa"),
    ("I am hungry in Delhi", "Delhi"),
    ("good food near Taj Mahal", "Taj Mahal"),
    ("find me restaurants in Kerala", "Kerala"),
    ("lunch spots in Rajasthan", "Rajasthan"),
    ("veg food in Goa", "Goa"),
]
for q, city in queries:
    s = new_conv()
    run(f"'{q}' → restaurant results", s, q,
        must_contain=["Restaurant", "Rating"], must_not=["You said"])

# ─── GROUP D: DISTANCE — DIFFERENT FORMATS ───────────────────
print(f"\n{B}─── D: Distance query variations ───{R}")
dist_queries = [
    "how far is Delhi from Agra",
    "distance between Goa and Kerala",
    "distance from Taj Mahal to India Gate",
    "how many km from Goa to Delhi",
    "how long does it take to get from Delhi to Agra",
]
for q in dist_queries:
    s = new_conv()
    run(f"'{q}' → km result", s, q, must_contain=["km"])

# ─── GROUP E: TRIP PLANNING — FULL CONVERSATIONS ─────────────
print(f"\n{B}─── E: Full trip planning flow ───{R}")

# E1: Standard flow
s = new_conv()
run("E1-1: 'I want to plan a trip' → asks destination", s,
    "I want to plan a trip", must_contain=["where", "travel"])
run("E1-2: 'Delhi' → asks days", s, "Delhi", must_contain=["days", "how many"])
run("E1-3: '5 days' → generates itinerary (budget now optional)", s, "5 days", must_contain=["Day", "Visit"])

# E2: One-shot trip request
s = new_conv()
run("E2: 'plan a 3 day trip to Goa' → immediate itinerary", s,
    "plan a 3 day trip to Goa", must_contain=["Day"])

# E3: Natural language destination
s = new_conv()
run("E3-1: 'I want to go to Rajasthan' → asks days", s,
    "I want to go to Rajasthan", must_contain=["days", "how many"])
run("E3-2: 'two days' → generates plan immediately (budget optional)", s,
    "two days", must_contain=["Day", "Visit"])

# ─── GROUP F: INTENT DETECTION — TRICKY PHRASES ──────────────
print(f"\n{B}─── F: Tricky intent detection (real-world edge cases) ───{R}")

# F1: Words that could confuse the matcher
tricky = [
    ("hotels in delhi", "hotel"),
    ("restaurants near india gate", "restaurant"),
    ("how far is the taj mahal from delhi", "distance"),
    ("i need a hotel in goa", "hotel"),
    ("show me places to visit in kerala", "location/plan"),
    ("what are things to do in goa", "location/plan"),
    ("i am starving near agra fort", "restaurant"),
    ("need a room in jaipur", "hotel"),
    ("suggest places in rajasthan", "location/plan"),
]
for q, intent_type in tricky:
    s = new_conv()
    resp = get_response(q, s)
    print(f"\n{CY}Q [{intent_type}]: {q}{R}")
    print(f"A: {resp[:250]}{'...' if len(resp)>250 else ''}")
    # Just check it's not a totally wrong response
    ok = resp.strip() and "You said" not in resp
    ok_text = f"{G}PASS{R}" if ok else f"{RED}FAIL{R}"
    print(f"  {B}{ok_text}{R}  Response not empty / not placeholder")
    if ok: passed+=1
    else:  failed+=1

# ─── GROUP G: RESET / RESTART FLOWS ──────────────────────────
print(f"\n{B}─── G: Reset and restart ───{R}")
s = new_conv()
get_response("plan a trip", s); get_response("Goa", s)
run("G1: 'start over' mid-flow → resets", s, "start over",
    must_contain=["reset", "travel"])

s = new_conv()
get_response("plan a trip", s); get_response("Delhi", s); get_response("3 days", s)
run("G2: 'forget everything' → resets", s, "forget everything",
    must_contain=["reset"])

# ─── GROUP H: CONTEXT SWITCH ──────────────────────────────────
print(f"\n{B}─── H: Context switch mid-planning ───{R}")
s = new_conv()
get_response("plan a trip", s); get_response("Goa", s)
run("H1: 'change destination' → clears destination", s,
    "change destination", must_contain=["travel", "where"])

# ─── GROUP I: GOODBYE / OUT-OF-SCOPE ──────────────────────────
print(f"\n{B}─── I: Goodbye & out-of-scope ───{R}")
for phrase in ["bye", "goodbye", "thanks bye", "see you", "take care"]:
    s = new_conv()
    run(f"'{phrase}' → farewell", s, phrase, must_contain=["goodbye", "safe"])

s = new_conv()
run("Out-of-scope: 'tell me a joke'", s, "tell me a joke",
    must_not=["Day 1", "km", "Hotel"])

# ─── GROUP J: LANDMARK-SPECIFIC QUERIES ───────────────────────
print(f"\n{B}─── J: Famous India landmark queries ───{R}")
landmarks = [
    "places to visit near Taj Mahal",
    "hotels near Golden Temple",
    "restaurants near Mysore Palace",
    "things to do near Hawa Mahal",
    "places near Red Fort",
]
for q in landmarks:
    s = new_conv()
    resp = get_response(q, s)
    print(f"\n{CY}Q: {q}{R}")
    print(f"A: {resp[:200]}{'...' if len(resp)>200 else ''}")
    ok = bool(resp.strip()) and "You said" not in resp
    print(f"  {B}{G if ok else RED}{'PASS' if ok else 'FAIL'}{R}{R}  Returns a valid response")
    if ok: passed+=1
    else:  failed+=1

# ─── SUMMARY ──────────────────────────────────────────────────
total = passed + failed
print(f"\n{'='*65}")
print(f"  {B}RESULTS: {G}{passed} passed{R}{B}, {RED}{failed} failed{R}{B} / {total} total{R}")
print(f"{'='*65}\n")
