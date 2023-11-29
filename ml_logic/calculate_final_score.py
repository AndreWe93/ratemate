import pandas as pd
import numpy as np


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

def df_with_score(df, price_weight, service_weight, atmosphere_weight, food_weight):
    df['average_score'] = df.apply(calculate_average_score,
    args=(price_weight, service_weight, atmosphere_weight, food_weight),
    axis=1
    )
    return df

def overall_score(df):
    return df.average_score.mean()
