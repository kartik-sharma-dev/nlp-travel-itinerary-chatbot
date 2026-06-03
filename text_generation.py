def nearby_location(locations_list):
    previous_location = None

    for location, distance in locations_list.items():
        if previous_location is None:
            print(f"You can go to {location}, which is {distance} km away.")
        else:
            print(f"Then you can go to {location}, which is {distance} km away from {previous_location}.")

        previous_location = location


def distance_between_two(distance,source,destination):
    return f"The distance between two {source} and {destination} is {distance}"


locations = {
    "A": 2,
    "B": 3,
    "C": 31,
    "D":2
}
nearby_location(locations)    
