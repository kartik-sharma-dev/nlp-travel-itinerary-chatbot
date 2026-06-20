import math
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

LABELS = ['A', 'B', 'C', 'D', 'E']

def calculate_similarity(query, target):
    try:
        if not query or not target:
            return 0
        query_tokens  = query.split()
        target_tokens = target.split()
        scores = []
        for t in target_tokens:
            if not query_tokens:
                scores.append(0)
                continue
            best_q = max(query_tokens, key=lambda q: fuzz.ratio(t, q))
            raw    = fuzz.ratio(t, best_q)
            # squared length ratio so short shared substrings don't inflate the score
            max_len = max(len(t), len(best_q))
            min_len = min(len(t), len(best_q))
            lr2 = (min_len / max_len) ** 2 if max_len > 0 else 0
            scores.append(raw * lr2)
        return sum(scores) / len(scores) if scores else 0
    except Exception as e:
        print(f"Error in calculate_similarity: {e}")
        return 0


def haversine(lat1, lon1, lat2, lon2):
    try:
        R = 6371
        lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = math.sin(dlat / 2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2)**2
        return round(R * 2 * math.asin(math.sqrt(a)), 2)
    except Exception as e:
        print(f"Error in haversine: {e}")
        return 0


def next_closest(lat, lon, nearby, visited_coords, min_distance=0):
    try:
        col       = 'dist_tmp'
        distances = []
        for _, row in nearby.iterrows():
            coord = (row['lat_location'], row['lon_location'])
            if coord in visited_coords:
                distances.append(None)
            else:
                distances.append(haversine(lat, lon, row['lat_location'], row['lon_location']))
        nearby      = nearby.copy()
        nearby[col] = distances
        valid       = nearby.dropna(subset=[col])
        valid       = valid[valid[col] > min_distance]
        if valid.empty:
            return None, None
        best_idx = valid[col].idxmin()
        return valid.loc[best_idx], valid.loc[best_idx, col]
    except Exception as e:
        print(f"Error in next_closest: {e}")
        return None, None


def score_and_rank(processed, data, vectorizer, tfidf_matrix):
    try:
        def fuzzyscore(row):
            place   = calculate_similarity(processed, row['proc_location']) * 0.70
            state   = calculate_similarity(processed, row['proc_state'])    * 0.20
            country = calculate_similarity(processed, row['proc_country'])  * 0.10
            return place + state + country

        scored                = data.copy()
        scored['fuzz_score']  = scored.apply(fuzzyscore, axis=1)
        query_vec             = vectorizer.transform([processed])
        cosine_scores         = cosine_similarity(query_vec, tfidf_matrix).flatten() * 100
        scored['tfidf_score'] = cosine_scores
        scored['final_score'] = scored['fuzz_score'] * 0.7 + scored['tfidf_score'] * 0.3
        return scored.sort_values('final_score', ascending=False)
    except Exception as e:
        print(f"Error in score_and_rank: {e}")
        return data


def build_chain(best_row, data, max_stops=None, min_distance=0, place_only=False, blacklist=None):
    try:
        if max_stops is None:
            max_stops = len(LABELS)
        if blacklist is None:
            blacklist = set()

        state_name = best_row['state']
        nearby     = data[
            (data['state'] == state_name) &
            data['lat_location'].notna() &
            data['lon_location'].notna()
        ].copy()

        if place_only:
            nearby = nearby[nearby['type'] == 'place']

        chain          = [best_row]
        visited_coords = {(best_row['lat_location'], best_row['lon_location'])}
        # pre-seed with blacklist so rejected places are skipped from the start
        visited_names  = {best_row['landmark']}.union(blacklist)

        for _ in range(max_stops - 1):
            prev       = chain[-1]
            candidates = nearby[~nearby['landmark'].isin(visited_names)]
            row, _     = next_closest(
                prev['lat_location'], prev['lon_location'],
                candidates, visited_coords, min_distance=min_distance
            )
            if row is None:
                break
            chain.append(row)
            visited_coords.add((row['lat_location'], row['lon_location']))
            visited_names.add(row['landmark'])

        return chain
    except Exception as e:
        print(f"Error in build_chain: {e}")
        return [best_row]
