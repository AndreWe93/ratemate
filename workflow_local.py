import pandas as pd

from ml_logic.text_preprocessor import TextPreprocessor
from ml_logic.bert_nlp_classification import process_reviews
from ml_logic.calculate_final_score import *

# Load the dataset
print("Loading data...")

# Take user input for the file path with a default value
default_file_path = "./raw_data_slim/merged_slim_file.csv"
file_path = input(f"default dataset with path: '{default_file_path}' if not enter the dataset ") or default_file_path

raw_dataset = pd.read_csv(file_path)
print("Data loaded ✅")

# Clean the dataset
print("Cleaning data...")
text_preprocessor = TextPreprocessor(file_path)
text_preprocessor.preprocess_dataset()
whole_dataset = text_preprocessor.google_reviews_df
print(whole_dataset.head(3))

try:
    # Assuming you have a larger dataset or loading it from a file
    larger_dataset = whole_dataset.copy()

    # Check if the dataset is not empty
    if larger_dataset is not None and not larger_dataset.empty:
        # Create a smaller dataset with 10 rows
        smaller_dataset = larger_dataset.head(10)

        # Display the smaller dataset
        print(smaller_dataset)
        print("Data cleaned ✅")
        print(smaller_dataset.head(3))
    else:
        print("Error: Unable to load or empty dataset.")
except Exception as e:
    print(f"Error: {e}")


# Classify the reviews
print("Classifying reviews...")
df_small = whole_dataset.head(10)
result_df_small = process_reviews(df_small, "reviews_without_SW")

print(result_df_small)
print("Classified reviews ✅")


# Calculate the final score
print("Calculating final score...")
sub_ratings_df = fill_sub_ratings(result_df_small, only_price=False)
print(sub_ratings_df)

final_score_df = df_with_score(sub_ratings_df, price_weight=0.25, service_weight=0.25, atmosphere_weight=0.25, food_weight=0.25)
print(final_score_df)

ratemate_score = overall_score(final_score_df)
print(ratemate_score)
print("Final score calculated ✅")
