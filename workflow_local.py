import pandas as pd

from ml_logic.text_preprocessor import TextPreprocessor
from ml_logic.bert_nlp_classification import process_reviews
from ml_logic.calculate_final_score import *
from ml_logic.NLP import *



# Take user input for the file path or take the default path
default_file_path = "./raw_data_slim/merged_slim_file.csv"
file_path = input(f"Enter dataset with path, default value = : '{default_file_path}'") or default_file_path

# Load the dataset
print("Loading dataset...")
your_dataset = pd.read_csv(file_path)
print("Data loaded ✅")
print(your_dataset.head(3))

# Small dataset for testing
df_small = your_dataset.head(10).copy()

# Preprocess the reviews
print("Preprocessing dataset...")
text_preprocessor = TextPreprocessor(df_small)
text_preprocessor.preprocess_dataset()
processed_df_small = text_preprocessor.google_reviews_df
print("Done ✅")
print(processed_df_small.head(3))

# Classify the reviews
print("Classifying reviews...")
result_df_small = process_reviews(df_small, "reviews_without_SW")

print("Classified reviews ✅")
print(result_df_small.head(3))

# Calculate the final score

print("Calculating final score...")


sub_ratings_df = fill_sub_ratings(result_df_small, only_price=True)
try:
    sub_ratings_df = new_column_NLP(sub_ratings_df)
except:
    print("********DO 'save_models.py' first******")

print(sub_ratings_df.head(3))



final_score_df = df_with_score(sub_ratings_df, price_weight=0.25, service_weight=0.25, atmosphere_weight=0.25, food_weight=0.25)
print(f"Dataset with final scores {final_score_df.head(3)}")

ratemate_score = overall_score(final_score_df)
print("Final score calculated ✅")
print(f"Final score is {ratemate_score} ⭐️⭐️⭐️⭐️⭐️")
