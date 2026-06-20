import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocess import preprocess_batch

BASE       = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE, "..", "Data", "real_landmark_locations.csv")
CACHE_PATH = os.path.join(BASE, ".data_cache.pkl")


def load_all_data():
    # use cached version if the CSV hasn't changed since last run
    try:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            if cache["mtime"] == os.path.getmtime(CSV_PATH):
                print("Loading from cache...")
                return (cache["data"], cache["hotel_df"], cache["restaurant_df"],
                        cache["landmark_df"], cache["vectorizer"], cache["tfidf_matrix"])
    except Exception as e:
        print(f"Cache load failed, rebuilding from CSV: {e}")

    try:
        print("Preprocessing data (first run only, spaCy running on full dataset)...")
        data = pd.read_csv(CSV_PATH).fillna("")
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV: {e}")

    try:
        missing = {"location", "landmark", "state", "country",
                   "lon_location", "lat_location",
                   "type", "ratings", "total_reviews"} - set(data.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        data["proc_location"] = preprocess_batch(data["location"].tolist())
        data["proc_landmark"] = preprocess_batch(data["landmark"].tolist())
        data["proc_state"]    = preprocess_batch(data["state"].tolist())
        data["proc_country"]  = preprocess_batch(data["country"].tolist())
        data["combined"]      = (data["proc_location"] + " " +
                                 data["proc_landmark"]  + " " +
                                 data["proc_state"]     + " " +
                                 data["proc_country"])

        vectorizer   = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        tfidf_matrix = vectorizer.fit_transform(data["combined"])

        hotel_df      = data[data["type"] == "hotel"].reset_index(drop=True)
        restaurant_df = data[data["type"] == "restaurant"].reset_index(drop=True)
        landmark_df   = data[data["type"] == "landmark"].reset_index(drop=True)
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error during preprocessing: {e}")

    # save so the next startup is instant
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({
                "mtime"        : os.path.getmtime(CSV_PATH),
                "data"         : data,
                "hotel_df"     : hotel_df,
                "restaurant_df": restaurant_df,
                "landmark_df"  : landmark_df,
                "vectorizer"   : vectorizer,
                "tfidf_matrix" : tfidf_matrix,
            }, f)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return data, hotel_df, restaurant_df, landmark_df, vectorizer, tfidf_matrix
