import pandas as pd

from ml_logic.text_preprocessor import TextPreprocessor
from ml_logic.bert_nlp_classification import process_reviews
from ml_logic.calculate_final_score import *

def load_dataset(file_path="./raw_data_slim/merged_slim_file.csv"):
    return pd.read_csv(file_path)

def scrape_apify(url):
    return dataframe

def preprocess_reviews_text(df):
    '''
    1. Creates new column with all the english reviews (Combination of #Text and #TranslatedText)
    2. NLP Preprocessing
    2.1 Creates a new column with NLP preprocessed text with Stopwords
    2.2 Creates a new column with NLP preprocessed text without Stopwords
    3. Keeps all the review columns except for the new NLP preprocessed columns
    '''
    text_preprocessor = TextPreprocessor(df)
    text_preprocessor.preprocess_dataset()
    processed_df = text_preprocessor.google_reviews_df
    return processed_df

def classify_reviews_df(df, column):
    return process_reviews(df, column)

def create_sub_ratings(df, only_price=False):
    return fill_sub_ratings(df, only_price=False)

def calculate_average_scores(df, price_weight, service_weight, atmosphere_weight, food_weight):
    return df_with_score(df, price_weight, service_weight, atmosphere_weight, food_weight)

def calculate_overall_score(df):
    return overall_score(df)


if __name__ == "__main__":
    user_df = scrape_apify(url)
    print("User dataset scrapped and loaded ✅")

    # Feature selection (check column names in scrapped data)

    processed_df = preprocess_reviews_text(user_df)
    print("Preprocessed dataset ✅")

    classify_reviews_df_small = classify_reviews_df(processed_df_small, "reviews_without_SW")
    # classify_reviews_df = process_reviews(processed_df, "reviews_without_SW")
    print("Classified reviews ✅")

    # Load the production model and calculate subratings (without the price subrating)
    # Fill n the price subrating after figuring it out
    # In the end don't use fill_sub_ratings function

    # Fine tune the following steps
    average_scores_df_small = calculate_average_scores(subratings_df_small, price_weight=0.25, service_weight=0.25, atmosphere_weight=0.25, food_weight=0.25)
    # average_scores_df = calculate_average_scores(subratings_df, price_weight=0.25, service_weight=0.25, atmosphere_weight=0.25, food_weight=0.25)
    print("Calculated average scores ✅")

    overall_score_small = calculate_overall_score(average_scores_df_small)
    # overall_score = calculate_overall_score(average_scores_df)
    print("Calculated overall score ✅")
    print(f"⭐️ Final score is {overall_score_small} ⭐️")
    pass
