import math
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

LABELS = ['A', 'B', 'C', 'D', 'E']

try:
    def calculate_similarity(query, target):
        if not query or not target:
            return 0
        query_tokens  = query.split()
        target_tokens = target.split()
        scores = []
        for t in target_tokens:
            bestmatch = max([fuzz.ratio(t, q) for q in query_tokens]) if query_tokens else 0
            scores.append(bestmatch)
        return sum(scores) / len(scores) if scores else 0

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = math.sin(dlat / 2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2)**2
        return round(R * 2 * math.asin(math.sqrt(a)), 2)

    def next_closest(lat, lon, nearby, visited_coords, min_distance=0):
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

    def score_and_rank(processed, data, vectorizer, tfidf_matrix):
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

    def build_chain(best_row, data, max_stops=None, min_distance=0, place_only=False):
        if max_stops is None:
            max_stops = len(LABELS)
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
        visited_names  = {best_row['landmark']}
        for _ in range(max_stops - 1):
            prev        = chain[-1]
            candidates  = nearby[~nearby['landmark'].isin(visited_names)]
            row, _      = next_closest(
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
    print(f"Error: {e}")