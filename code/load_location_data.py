import os
import pandas as pd
from preprocess import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer


def load_location_data():
    base = os.path.dirname(os.path.abspath(__file__))

    data1 = pd.read_csv(os.path.join(base, 'location.csv'))

    data2 = pd.read_csv(os.path.join(base, 'hotel_places_directory.csv'))
    data2 = data2.rename(columns={
        'country': 'Country',
        'state': 'State/Province',
        'place name': 'PlaceName',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
    })

    data3 = pd.read_csv(os.path.join(base, 'hotel_tourism_plans.csv'))
    data3 = data3.rename(columns={
        'state': 'State/Province',
        'place name': 'PlaceName',
    })

    data4 = pd.read_csv(os.path.join(base, 'places_and_restaurants.csv'))
    data4 = data4.rename(columns={'Place Name': 'PlaceName'})

    data = pd.concat([data1, data2, data3, data4], ignore_index=True)
    data.drop_duplicates(inplace=True)
    data = data.fillna('')
    data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
    data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
    data['processed_country'] = data['Country'].apply(preprocess)
    data['processed_state'] = data['State/Province'].apply(preprocess)
    data['processed_place_name'] = data['PlaceName'].apply(preprocess)
    data['combined'] = (
        data['processed_place_name'] + " " +
        data['processed_state'] + " " +
        data['processed_country']
    )

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    tfidf_matrix = vectorizer.fit_transform(data['processed_place_name'])

    return data, vectorizer, tfidf_matrix


def load_tourism_data():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hotel_tourism_plans.csv')
    data = pd.read_csv(filepath)
    data = data.rename(columns={
        'state': 'State/Province',
        'place name': 'PlaceName',
    })
    data = data.fillna('')
    data['processed_state'] = data['State/Province'].apply(preprocess)
    data['processed_place_name'] = data['PlaceName'].apply(preprocess)
    return data


def load_restaurant_data():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'places_and_restaurants.csv')
    data = pd.read_csv(filepath)
    data = data.rename(columns={'Place Name': 'PlaceName'})
    data = data.fillna('')
    data['processed_country'] = data['Country'].apply(preprocess)
    data['processed_state'] = data['State/Province'].apply(preprocess)
    data['processed_place_name'] = data['PlaceName'].apply(preprocess)
    return data


def loaddata():
    data, vectorizer, tfidf_matrix = load_location_data()
    tourism_data = load_tourism_data()
    restaurant_data = load_restaurant_data()
    return data, tourism_data, restaurant_data, vectorizer, tfidf_matrix
