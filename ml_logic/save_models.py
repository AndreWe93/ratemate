from ml_logic.NLP import pretraining_NLP_models, get_dataset_for_NLP, get_Xy_for_NLP,in_production
import pandas as pd


# Take user input for the file path or take the default path
default_file_path = "./raw_data_slim/merged_slim_file.csv"

path = "./raw_data_slim/Pepenero Schwabing.csv" #just for Katya

file_path = input(f"Enter full dataset for training with path, default value = : '{default_file_path}'") or default_file_path

# Load the dataset
print("Loading dataset...")

your_dataset = pd.read_csv(file_path)

print("Data loaded ✅")

print("Training and saving models..")

data = get_dataset_for_NLP(your_dataset)
X, y = get_Xy_for_NLP(data)

pretraining_NLP_models(X,y)

print("All models saved to MLFlow ✅")




in_production()
