"""
Q&A test runner for the travel chatbot.
Run: python test_bot.py
"""
import sys, os

USER_DIR = os.path.join(os.path.dirname(__file__), "user")
sys.path.insert(0, USER_DIR)

from nlp_bridge import get_response, _new_session

RESET = "\033[0m"
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

def new_conv():
    """Start a fresh conversation session."""
    return _new_session()

def ask(session, question, label=""):
    resp = get_response(question, session)
    tag = f" [{label}]" if label else ""
    print(f"\n{CYAN}Q{tag}: {question}{RESET}")
    print(f"A: {resp}")
    return resp

def check(desc, resp, must_contain=None, must_not_contain=None, non_empty=True):
    ok = True
    reasons = []
    if non_empty and not resp.strip():
        ok = False; reasons.append("response is empty")
    if must_contain:
        for kw in must_contain:
            if kw.lower() not in resp.lower():
                ok = False; reasons.append(f"missing '{kw}'")
    if must_not_contain:
        for kw in must_not_contain:
            if kw.lower() in resp.lower():
                ok = False; reasons.append(f"should NOT contain '{kw}'")
    status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    reason_str = f"  {DIM}({', '.join(reasons)}){RESET}" if reasons else ""
    print(f"  {BOLD}{status}{RESET}  {desc}{reason_str}")
    return ok

pass_count = 0
fail_count = 0

def run(desc, session, question, must_contain=None, must_not_contain=None, label=""):
    global pass_count, fail_count
    resp = ask(session, question, label)
    ok = check(desc, resp, must_contain, must_not_contain)
    if ok: pass_count += 1
    else:  fail_count += 1
    return resp

print(f"\n{'='*60}")
print(f"  {BOLD}TRAVEL BOT Q&A TEST SUITE{RESET}")
print(f"{'='*60}")

# ─── GROUP 1: GREETING ────────────────────────────────────────
print(f"\n{BOLD}─── GROUP 1: Greetings ───{RESET}")
s = new_conv()
run("'hi' returns a greeting", s, "hi",
    must_not_contain=["You said"])

s = new_conv()
run("'hello' returns a greeting", s, "hello",
    must_not_contain=["You said"])

s = new_conv()
run("'help' returns capability list", s, "help me",
    must_contain=["hotel", "restaurant"])

# ─── GROUP 2: INTENT DETECTION (priority order) ───────────────
print(f"\n{BOLD}─── GROUP 2: Intent priority (the fixed bugs) ───{RESET}")
s = new_conv()
run("'restart the planning' triggers reset, NOT greeting", s, "restart the planning",
    must_contain=["reset", "travel"])

s = new_conv()
run("'go to a restaurant in Goa' triggers restaurant, NOT location", s, "restaurants in Goa",
    must_not_contain=["plan", "trip", "itinerary"])

s = new_conv()
run("'distance between Mumbai and Goa' returns km result", s, "distance between Mumbai and Goa",
    must_contain=["km"])

# ─── GROUP 3: DIRECT STANDALONE QUERIES ───────────────────────
print(f"\n{BOLD}─── GROUP 3: Standalone queries (hotel / restaurant / distance) ───{RESET}")
s = new_conv()
run("Hotels near Jaipur returns hotel results", s, "hotels near Jaipur",
    must_not_contain=["You said", "couldn't find"])

s = new_conv()
run("Restaurants in Mumbai returns results", s, "restaurants in Mumbai",
    must_not_contain=["You said", "couldn't find"])

s = new_conv()
run("Distance Delhi to Agra returns numeric km", s, "how far is Delhi from Agra",
    must_contain=["km"])

# ─── GROUP 4: TRIP PLANNING FLOW ──────────────────────────────
print(f"\n{BOLD}─── GROUP 4: Trip planning conversation ───{RESET}")
s = new_conv()
r1 = run("Step 1 - trigger planning", s, "plan a trip",
         must_contain=["travel", "where"])

r2 = run("Step 2 - give destination", s, "Goa",
         must_contain=["days", "how many", "stay"])

r3 = run("Step 3 - give days → bot generates plan (budget optional)", s, "3 days",
         must_contain=["Day", "Visit"])

# ─── GROUP 5: DESTINATION EXTRACTION FIX ──────────────────────
print(f"\n{BOLD}─── GROUP 5: Destination extraction (fixed bug) ───{RESET}")
s = new_conv()
s["state"] = "collecting_info"
r = run("'I want to go to Jaipur' → stores 'Jaipur' not full sentence", s,
        "I want to go to Jaipur",
        label="destination fill")
dest = s["collected"].get("destination", "")
ok_dest = dest and len(dest) < 20 and "want" not in dest.lower()
print(f"  {GREEN if ok_dest else RED}{'PASS' if ok_dest else 'FAIL'}{RESET}  "
      f"Destination stored as '{dest}' (should be short city name)")
if ok_dest: pass_count += 1
else:        fail_count += 1

# ─── GROUP 6: SESSION ISOLATION ───────────────────────────────
print(f"\n{BOLD}─── GROUP 6: Per-user session isolation ───{RESET}")
s1 = new_conv()
s2 = new_conv()
get_response("plan a trip", s1)
get_response("Goa", s1)
# s2 should still be fresh
state2 = s2.get("state", "greeting")
dest2  = s2["collected"].get("destination")
ok_iso = state2 == "greeting" and dest2 is None
print(f"  {GREEN if ok_iso else RED}{'PASS' if ok_iso else 'FAIL'}{RESET}  "
      f"Session 2 unaffected by Session 1 (state='{state2}', dest='{dest2}')")
if ok_iso: pass_count += 1
else:        fail_count += 1

# ─── GROUP 7: STATE MACHINE EDGE CASES ───────────────────────
print(f"\n{BOLD}─── GROUP 7: State machine edge cases ───{RESET}")
s = new_conv()
s["state"] = "reviewing"
s["collected"]["destination"] = "Goa"
run("Reviewing state: 'yes' finalizes the trip", s, "yes",
    must_contain=["wonderful", "trip", "Goa"])

s = new_conv()
s["state"] = "finalized"
s["collected"]["destination"] = "Goa"
run("Finalized state: 'hi' offers new trip plan", s, "hi",
    must_contain=["trip", "next", "destination", "visit"])

s = new_conv()
run("'goodbye' returns farewell message", s, "bye",
    must_contain=["goodbye", "safe", "trip"])

# ─── SUMMARY ──────────────────────────────────────────────────
total = pass_count + fail_count
print(f"\n{'='*60}")
print(f"  {BOLD}RESULTS: {GREEN}{pass_count} passed{RESET}{BOLD}, {RED}{fail_count} failed{RESET}{BOLD} / {total} total{RESET}")
print(f"{'='*60}\n")
