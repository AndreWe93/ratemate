from ml_logic.NLP import pretraining_NLP_models, get_dataset_for_NLP, get_Xy_for_NLP
import pandas as pd


# Take user input for the file path or take the default path
default_file_path = "./raw_data_slim/merged_slim_file.csv"

path = "Ratemate/raw_data/merged_slim_file.csv" #just for Katya

file_path = input(f"Enter full dataset for training with path, default value = : '{default_file_path}'") or default_file_path

# Load the dataset
print("Loading dataset...")
your_dataset = pd.read_csv(file_path)
print("Data loaded ✅")
data, df_full = get_dataset_for_NLP(your_dataset)
X, y = get_Xy_for_NLP(data)
pretraining_NLP_models(X,y)
