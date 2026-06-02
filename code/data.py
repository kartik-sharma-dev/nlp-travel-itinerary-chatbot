import pandas as pd
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent

df = pd.read_csv(BASE_DIR / 'hotel_places_directory.csv')

# Prepare the final dataframe with your exact requested column names
columns = [
    'country', 'state', 'place name', 'hotel-name', 'rating', 
    'person aloud per room', 'food avalabel or not', 'taxi avalabel or not', 
    'room type', 'price', 'hotel_longitude', 'hotel_latitude', 
    'place_longitude', 'place_latitude'
]

new_rows = []

# Process each row
for index, row in df.iterrows():
    place_lon = row['longitude']
    place_lat = row['latitude']
    
    hotel_name = str(row['hotel-name']).lower()
    
    # Create realistic, deterministic offsets based on hotel type so they stay consistent
    if 'grand' in hotel_name:
        lat_offset = 0.0023
        lon_offset = 0.0124
    elif 'premium' in hotel_name:
        lat_offset = -0.0171
        lon_offset = -0.0067
    else:
        lat_offset = 0.0096
        lon_offset = 0.0060

    hotel_lat = round(place_lat + lat_offset, 4)
    hotel_lon = round(place_lon + lon_offset, 4)
    
    new_rows.append({
        'country': row['country'],
        'state': row['state'],
        'place name': row['place name'],
        'hotel-name': row['hotel-name'],
        'rating': row['rating'],
        'person aloud per room': row['person aloud per room'],
        'food avalabel or not': row['food avalabel or not'],
        'taxi avalabel or not': row['taxi avalabel or not'],
        'room type': row['room type'],
        'price': row['price'],
        'hotel_longitude': hotel_lon,
        'hotel_latitude': hotel_lat,
        'place_longitude': place_lon,
        'place_latitude': place_lat
    })

# Create the new dataframe and export to CSV
final_df = pd.DataFrame(new_rows, columns=columns)
final_df.to_csv(BASE_DIR / 'final_expanded_dataset.csv', index=False)
print("Complete! Check final_expanded_dataset.csv")