import pandas as pd

from ml_logic.text_preprocessor import TextPreprocessor
from ml_logic.bert_nlp_classification import process_reviews
from ml_logic.calculate_final_score import *
from ml_logic.random_forest_model import pred_from_random_forest


# Take user input for the file path or take the default path
default_file_path = "./raw_data_slim/merged_slim_file.csv"
# default_file_path = "./raw_data_slim/merged_thai_restaurant_file.csv"
file_path = input(f"Enter dataset with path, default value = : '{default_file_path}'") or default_file_path

# Load the dataset
print("Loading dataset...")
your_dataset = pd.read_csv(default_file_path)
print("Data loaded ✅")
print(your_dataset.head(3))

# Preprocess the reviews
print("Preprocessing dataset...")
text_preprocessor = TextPreprocessor(your_dataset)
text_preprocessor.preprocess_dataset()
processed_df = text_preprocessor.google_reviews_df
print("Done ✅")
print(processed_df.head(3))

# Classify the reviews
print("Classifying reviews...")
result_df = process_reviews(processed_df, "reviews_without_SW")
print("Classified reviews ✅")
print(result_df.head(3))

# Calculate the final score
subratings_df = pred_from_random_forest(result_df)
subratings_df_price = df_with_price_rating(subratings_df)
print("Created subratings ✅")

average_scores_df = df_with_score(subratings_df_price, price_weight=0.25, service_weight=0.25, atmosphere_weight=0.25, food_weight=0.25)
print("Calculated average scores ✅")

print(f"Dataset with final scores {average_scores_df.head(3)}")

ratemate_score = overall_score(average_scores_df)
print("Final score calculated ✅")
print(f"Final score is {ratemate_score} ⭐️⭐️⭐️⭐️⭐️")
