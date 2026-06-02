from load_location_data import loaddata
from detect_intent import detect_intent, update_intent, wrong_intent
from handlers import handle_query, handle_distance_query, location, handle_greeting

data, tourism_data, restaurant_data, vectorizer, tfidf_matrix = loaddata()

if __name__ == '__main__':
    current_intent = None
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
        if not query:
            continue

        intent = detect_intent(query)
        intent = update_intent(current_intent, query)
        current_intent = intent
        print(f"Detected intent: {intent}")

        
        if intent == 'hotel':
            handle_query(query, data, vectorizer, tfidf_matrix)
        elif intent == 'distance':
            handle_distance_query(query, data, vectorizer, tfidf_matrix)
        elif intent == 'location':
            hotel_words = ['hotel', 'room', 'stay', 'accommodation', 'lodge', 'resort']
            if any(w in query.lower() for w in hotel_words):
                handle_query(query, data, vectorizer, tfidf_matrix)
            else:
                location(query, data, vectorizer, tfidf_matrix)
        elif intent == 'greeting':
            handle_greeting(query)
        elif intent is None:
            print("I'm here to help with hotels, restaurants, taxis, and nearby places. What would you like to know?")
        else:
            wrong_intent(intent)
