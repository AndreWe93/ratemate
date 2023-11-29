import pandas as pd

from ml_logic.text_preprocessor import TextPreprocessor
from ml_logic.bert_nlp_classification import process_reviews

print("Loading data...")

# Take user input for the file path with a default value
default_file_path = "./raw_data_slim/merged_slim_file.csv"
file_path = input(f"default dataset with path: '{default_file_path}' if not enter the dataset ") or default_file_path

raw_dataset = pd.read_csv(file_path)
print(raw_dataset.head(3))
print("Data loaded ✅")


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

print("Classifying reviews...")
df_small = whole_dataset.head(10)
result_df_small = process_reviews(df_small, "reviews_without_SW")
print(result_df_small)
print("Classified reviews ✅")
