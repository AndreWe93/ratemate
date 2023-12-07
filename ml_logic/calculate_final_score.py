import pandas as pd
import numpy as np


def fill_sub_ratings(df, only_price = False):
    if only_price is True:
        df["price_rating"] = np.random.randint(1, 6, size=len(df))
        return df
    else:
        df["price_rating"] = np.random.randint(1, 6, size=len(df))
        df["service_rating"] = np.random.randint(1, 6, size=len(df))
        df["atmosphere_rating"] = np.random.randint(1, 6, size=len(df))
        df["food_rating"] = np.random.randint(1, 6, size=len(df))
        return df


def calculate_price_subrating(row):
    # Step 1: Calculate the average of service, food, and atmosphere
    average_sfa = (row['service_rating'] + row['food_rating'] + row['atmosphere_rating']) / 3

    # Step 2: Calculate the price subrating
    price_subrating = max(1, min(2 * row['stars'] - average_sfa, 5))

    return price_subrating

def df_with_price_rating(df):
    df["price_rating"] = df.apply(calculate_price_subrating, axis = 1)

    return df

def calculate_average_score(row, price_weight, service_weight, atmosphere_weight, food_weight):
    # Explicitly reference the desired columns for rating
    price_rating = row['price_rating']
    service_rating = row['service_rating']
    atmosphere_rating = row['atmosphere_rating']
    food_rating = row['food_rating']

    # Multiply each rating by its corresponding weight and calculate the weighted sum
    weighted_sum = (
        price_rating * price_weight +
        service_rating * service_weight +
        atmosphere_rating * atmosphere_weight +
        food_rating * food_weight
    )

    # Calculate the weighted average score
    total_weight = price_weight + service_weight + atmosphere_weight + food_weight
    average_score = weighted_sum / total_weight

    return average_score

def calculate_average_score_class(row, price_weight, service_weight, atmosphere_weight, food_weight):
    # Explicitly reference the desired columns for rating
    price_rating = row['price_rating']
    service_rating = row['service_rating']
    atmosphere_rating = row['atmosphere_rating']
    food_rating = row['food_rating']

    price_class = row['price']
    service_class = row['service']
    atmosphere_class = row['atmosphere']
    food_class = row['food']

    # Multiply each rating by its corresponding weight and calculate the weighted sum
    weighted_sum = (
        price_rating * price_weight * price_class +
        service_rating * service_weight * service_class +
        atmosphere_rating * atmosphere_weight * atmosphere_class +
        food_rating * food_weight * food_class
    )

    # Calculate the weighted average score
    total_weight = (price_weight * price_class) + \
    (service_weight * service_class) + (atmosphere_weight * atmosphere_class)  + (food_weight * food_class)
    average_score_class = round((weighted_sum / total_weight),2)

    return average_score_class

def df_with_score(df, price_weight, service_weight, atmosphere_weight, food_weight, with_class = True):
    if with_class is True:
        df['average_score'] = df.apply(calculate_average_score_class,
        args=(price_weight, service_weight, atmosphere_weight, food_weight),
        axis=1
        )
        return df
    else:
        df['average_score'] = df.apply(calculate_average_score,
        args=(price_weight, service_weight, atmosphere_weight, food_weight),
        axis=1
        )
        return df

def overall_score(df):
    return round(df.average_score.mean(), 2)

def individual_scores(df):
    average_price = round(df.price_rating.mean(),2)
    average_service = round(df.service_rating.mean(),2)
    average_food = round(df.food_rating.mean(),2)
    average_atmosphere = round(df.atmosphere_rating.mean(),2)

    return average_price, average_service, average_atmosphere, average_food

def class_dist(df):
    dist_price = round(df["price"].mean().astype(np.float64),2)
    dist_service = round(df["service"].mean().astype(np.float64),2)
    dist_atmosphere = round(df["atmosphere"].mean().astype(np.float64),2)
    dist_food = round(df["food"].mean().astype(np.float64),2)

    return dist_price, dist_service, dist_atmosphere, dist_food
