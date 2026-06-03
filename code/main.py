from load_location_data import load_all_data
from detect_intent import detect_intent, update_intent, wrong_intent
from handlers import (
    handle_hotel_query,
    handle_restaurant_query,
    handle_distance_query,
    handle_location_query,
    handle_itinerary_query,
    handle_greeting,
)
# Load all data and models at startup to avoid repeated loading during queries
data, hotel_df, restaurant_df, landmark_df, vectorizer, tfidf_matrix = load_all_data()

if __name__ == '__main__':
    current_intent = None

    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()

        if query.lower() in ('quit', 'exit', 'q'):
            break

        if not query:
            continue

        intent         = detect_intent(query)
        current_intent = update_intent(current_intent, query)
        intent         = current_intent

        print(f"Detected intent: {intent}")

        if intent == 'greeting':
            handle_greeting(query)

        elif intent == 'hotel' or (isinstance(intent, dict) and intent.get('hotel')):
            handle_hotel_query(query, data, hotel_df, vectorizer, tfidf_matrix)

        elif intent == 'restaurant' or (isinstance(intent, dict) and intent.get('restaurant')):
            handle_restaurant_query(query, data, restaurant_df, vectorizer, tfidf_matrix)

        elif intent == 'itinerary':
            handle_itinerary_query(query, data, hotel_df, restaurant_df, vectorizer, tfidf_matrix)

        elif intent == 'distance':
            handle_distance_query(query, data, vectorizer, tfidf_matrix)

        elif intent == 'location':
            # if query also has hotel words, show hotels instead
            hotel_words = ['hotel', 'room', 'stay', 'accommodation', 'lodge', 'resort']
            if any(w in query.lower() for w in hotel_words):
                handle_hotel_query(query, data, hotel_df, vectorizer, tfidf_matrix)
            else:
                handle_location_query(query, data, vectorizer, tfidf_matrix)

        elif intent is None:
            print("I'm here to help with hotels, restaurants, and nearby places. What would you like to know?")

        else:
            wrong_intent(intent)