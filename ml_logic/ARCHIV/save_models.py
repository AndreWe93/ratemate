from ml_logic.NLP import training_NLP_combine_model_and_saving_to_mlflow, get_dataset_for_NLP, get_Xy_for_NLP,in_production
import pandas as pd


if __name__ == "__main__":
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
    X, y, X_num = get_Xy_for_NLP(data)

    training_NLP_combine_model_and_saving_to_mlflow(X,y,X_num)

    print("All models saved to MLFlow ✅")




    #in_production()
