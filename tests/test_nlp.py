"""
Tests for:
  - Intent detection on a handful of sample queries
  - 4-state session transition logic (greeting → collecting_info → reviewing → finalized)
  - Correction scenarios (reject a hotel, reject a day's activity)

Run from the project root:
    python -m pytest tests/ -v
or:
    python -m unittest discover -s tests -v
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Make `user` importable when running from the project root or from tests/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Pure-function imports (no ML model loading triggered) ──────────────────────
from user.nlp_engine.detect_intent import (
    detect_intent,
    session as new_session,
    get_missing_info,
    fill_slot,
    is_correction_query,
    parse_correction,
    handle_correction_followup,
)

# ── Patch load_all_data BEFORE nlp_bridge is imported ─────────────────────────
# nlp_bridge calls load_all_data() at module level; we replace it with six
# MagicMocks so no CSV or sklearn fitting happens during tests.
_FAKE_MODELS = tuple(MagicMock() for _ in range(6))
with patch("user.nlp_engine.load_location_data.load_all_data", return_value=_FAKE_MODELS):
    import user.nlp_bridge as _nlp_bridge        # noqa: E402
    from user.nlp_bridge import get_response     # noqa: E402


# ==============================================================================
# 1 – INTENT DETECTION
# ==============================================================================

class TestIntentDetection(unittest.TestCase):

    def test_hi_is_greeting(self):
        self.assertEqual(detect_intent("hi"), "greeting")

    def test_hello_there_is_greeting(self):
        self.assertEqual(detect_intent("hello there"), "greeting")

    def test_plan_trip_is_build_itinerary(self):
        self.assertEqual(detect_intent("plan a trip to Goa"), "build_itinerary")

    def test_week_holiday_is_build_itinerary(self):
        self.assertEqual(detect_intent("i want a week long holiday"), "build_itinerary")

    def test_distance_query(self):
        self.assertEqual(detect_intent("distance between Delhi and Mumbai"), "distance_query")

    def test_out_of_scope_joke(self):
        self.assertEqual(detect_intent("tell me a joke"), "out_of_scope")

    def test_deny_intent(self):
        self.assertEqual(detect_intent("no I don't want that"), "deny")

    def test_hungry_is_restaurant(self):
        self.assertEqual(detect_intent("I am hungry"), "restaurant")

    def test_yes_is_confirm(self):
        self.assertEqual(detect_intent("yes that looks great"), "confirm")

    def test_goodbye(self):
        self.assertEqual(detect_intent("bye thanks for your help"), "goodbye")


# ==============================================================================
# 2 – SLOT FILLING & get_missing_info
# ==============================================================================

class TestSlotFilling(unittest.TestCase):

    def setUp(self):
        self.sess = new_session()

    def test_fresh_session_missing_destination(self):
        self.assertEqual(get_missing_info(self.sess), "destination")

    def test_missing_days_after_destination_set(self):
        self.sess["collected"]["destination"] = "Goa"
        self.assertEqual(get_missing_info(self.sess), "days")

    def test_nothing_missing_when_both_set(self):
        self.sess["collected"]["destination"] = "Goa"
        self.sess["collected"]["days"] = 3
        self.assertIsNone(get_missing_info(self.sess))

    def test_fill_days_numeric_string(self):
        ok = fill_slot("days", "3 days", self.sess)
        self.assertTrue(ok)
        self.assertEqual(self.sess["collected"]["days"], 3)
        self.assertEqual(self.sess["trip_days"], 3)

    def test_fill_destination_titlecases_and_syncs(self):
        ok = fill_slot("destination", "goa", self.sess)
        self.assertTrue(ok)
        self.assertEqual(self.sess["collected"]["destination"], "Goa")
        self.assertEqual(self.sess["last_location"], "Goa")


# ==============================================================================
# 3 – STATE MACHINE TRANSITIONS
# ==============================================================================

class TestStateTransitions(unittest.TestCase):

    def _fresh(self):
        return new_session()

    def test_initial_state_is_greeting(self):
        self.assertEqual(self._fresh()["state"], "greeting")

    def test_greeting_message_stays_in_greeting(self):
        sess = self._fresh()
        response = get_response("hello", user_session=sess)
        self.assertEqual(sess["state"], "greeting")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_build_itinerary_intent_in_greeting_moves_to_collecting_info(self):
        """Sending a trip-planning query from greeting state should request the destination."""
        sess = self._fresh()
        response = get_response("I want to plan a trip", user_session=sess)
        self.assertEqual(sess["state"], "collecting_info")
        self.assertIn("travel", response.lower())

    @patch("user.nlp_bridge._run_itinerary", return_value="Here is your 3-day itinerary!")
    def test_collecting_info_completes_to_reviewing(self, mock_run):
        """Once destination + days are known, the bot generates an itinerary and enters reviewing."""
        sess = self._fresh()
        sess["state"] = "collecting_info"
        sess["collected"]["destination"] = "Goa"
        sess["last_location"] = "Goa"

        response = get_response("3 days", user_session=sess)

        self.assertEqual(sess["state"], "reviewing")
        self.assertIn("itinerary", response.lower())
        mock_run.assert_called_once()

    def test_confirm_in_reviewing_transitions_to_finalized(self):
        """'Yes' in reviewing state should lock the trip and move to finalized."""
        sess = self._fresh()
        sess["state"] = "reviewing"
        sess["collected"]["destination"] = "Goa"

        response = get_response("yes", user_session=sess)

        self.assertEqual(sess["state"], "finalized")
        self.assertIn("wonderful trip", response.lower())


# ==============================================================================
# 4 – CORRECTION SCENARIOS
# ==============================================================================

class TestCorrectionScenarios(unittest.TestCase):

    def setUp(self):
        self.sess = new_session()
        self.sess["state"] = "reviewing"
        self.sess["collected"]["destination"] = "Goa"
        self.sess["collected"]["days"] = 3
        # Simulate a generated plan so _run_correction can read it
        self.sess["generated_plan"] = {
            1: {
                "places": ["Fort Aguada"],
                "hotel":  "Taj Holiday Village",
                "rest":   "Martin's Corner",
            },
            2: {
                "places": ["Calangute Beach", "Vagator Beach"],
                "hotel":  "Park Hyatt Goa",
                "rest":   "Britto's",
            },
        }

    # -- is_correction_query --------------------------------------------------

    def test_swap_keyword_is_correction(self):
        self.assertTrue(is_correction_query("swap the hotel on day 1"))

    def test_dislike_phrase_is_correction(self):
        self.assertTrue(is_correction_query("I don't like that place"))

    def test_plain_greeting_is_not_correction(self):
        self.assertFalse(is_correction_query("hello how are you"))

    # -- parse_correction -----------------------------------------------------

    def test_parse_correction_hotel_day1(self):
        result = parse_correction("change the hotel on day 1")
        self.assertEqual(result["type"], "change_slot")
        self.assertEqual(result["day"], 1)
        self.assertEqual(result["slot"], "hotel")

    def test_parse_correction_morning_day2(self):
        result = parse_correction("replace the morning activity on day 2")
        self.assertEqual(result["type"], "change_slot")
        self.assertEqual(result["day"], 2)
        self.assertEqual(result["slot"], "morning")

    # -- _run_correction (blacklist logic) ------------------------------------

    @patch("user.nlp_bridge._run_itinerary", return_value="Updated itinerary!")
    def test_reject_hotel_adds_to_blacklist(self, mock_run):
        """Rejecting day-1 hotel should add that hotel name to rejected_hotels."""
        from user.nlp_bridge import _run_correction
        _run_correction({"type": "change_slot", "day": 1, "slot": "hotel"}, self.sess)
        self.assertIn("Taj Holiday Village", self.sess["rejected_hotels"])
        mock_run.assert_called_once()

    @patch("user.nlp_bridge._run_itinerary", return_value="Updated itinerary!")
    def test_reject_day2_restaurant_adds_to_blacklist(self, mock_run):
        """Rejecting day-2 dinner should add that restaurant to rejected_rests."""
        from user.nlp_bridge import _run_correction
        _run_correction({"type": "change_slot", "day": 2, "slot": "dinner"}, self.sess)
        self.assertIn("Britto's", self.sess["rejected_rests"])
        mock_run.assert_called_once()

    # -- handle_correction_followup (multi-turn dialog) -----------------------

    def test_followup_asks_for_day_when_missing(self):
        """If pending_correction has no day, the bot should ask which day."""
        self.sess["pending_correction"] = {"type": "change_slot", "slot": "hotel", "day": None}
        correction, question = handle_correction_followup("I want to change it", self.sess)
        self.assertIsNone(correction)
        self.assertIsNotNone(question)
        self.assertIn("day", question.lower())


if __name__ == "__main__":
    unittest.main()
