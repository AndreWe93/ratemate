from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ml_logic.tfdf import numeric_columns, text_column, y_columns, get_dataset_only_for_tfdf, new_columns_names
import pandas as pd
import mlflow
from params import *
from ml_logic.ran_forest import tfidf_vectorizer
import pickle
from ml_logic.registry import load_model, mlflow_transition_model



# Take user input for the file path or take the default path
default_file_path = "./raw_data_slim/merged_slim_file.csv"
#file_path = input(f"Enter full dataset for training with path, default value = '{default_file_path}'") or default_file_path

# Load the dataset
print("Loading dataset...")
your_dataset = pd.read_csv(default_file_path)
print("Data loaded ✅")



def run_load_CatBoost(your_dataset):
    print("Preparing dataset...")
    data = get_dataset_only_for_tfdf(your_dataset)
    print("Dataset prepared ✅")

    y = data[y_columns]
    text = data[text_column]
    numeric = data[numeric_columns]

    print("X, y prepared ✅")

    # Convert text to TF-IDF representation


    X_tfidf = tfidf_vectorizer(text)


    # Combine text and numeric data
    X = pd.concat([pd.DataFrame(X_tfidf.toarray()), numeric.reset_index(drop=True)], axis=1)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training and saving models..")

    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name='ratemate_catboost')

    with mlflow.start_run() as run:
        learning_rate = 0.4
        iterations=150
        #depth=15
        model_output1 = CatBoostClassifier(#learning_rate=learning_rate,
                                        #iterations= iterations,
                                        #depth= depth
                                        )
        model_output2 = CatBoostClassifier(#learning_rate=learning_rate,
                                        #iterations= iterations,
                                        #depth= depth
                                        )
        model_output3 = CatBoostClassifier(#learning_rate=learning_rate,
                                        #iterations= iterations,
                                        #depth= depth
                                        )


        model_output1.fit(X_train, y_train['reviewDetailedRating/Atmosphere'])
        model_output2.fit(X_train, y_train['reviewDetailedRating/Food'])
        model_output3.fit(X_train, y_train['reviewDetailedRating/Service'])


        # print(type(y_train['reviewDetailedRating/Atmosphere']), y_train['reviewDetailedRating/Atmosphere'].shape)
        # print(type(y_train['reviewDetailedRating/Food']), y_train['reviewDetailedRating/Food'].shape)


        # Make predictions
        pred_output1 = model_output1.predict(X_test)
        pred_output2 = model_output2.predict(X_test)
        pred_output3 = model_output3.predict(X_test)



        predictions = pd.DataFrame({
            'atmosphere_rating': pd.Series(pred_output1.flatten()),
            'food_rating': pd.Series(pred_output2.flatten()),
            'service_rating': pd.Series(pred_output3.flatten())
        })
        # Compute and print metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        mae_1 = mean_absolute_error(y_test['reviewDetailedRating/Atmosphere'],
                                    predictions['atmosphere_rating'])
        mae_2 = mean_absolute_error(y_test['reviewDetailedRating/Food'],
                                    predictions['food_rating'])
        mae_3 = mean_absolute_error(y_test['reviewDetailedRating/Service'],
                                    predictions['service_rating'])
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MAE_Atm", mae_1)
        mlflow.log_metric("MAE_Food", mae_2)
        mlflow.log_metric("MAE_Serv", mae_3)



        # Log the model
        for i, model in enumerate([model_output1, model_output2, model_output3]):

            params = model.get_params()
            mlflow.log_params(params)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"model_{i}",
                registered_model_name=f"catboost_classif{i}"
            )
    print('********done*******')


def new_column_CatBoost(your_dataset, save_local_csv=False):

    """
    adding new columns to df
    """
    text = your_dataset[text_column]
    numeric = your_dataset[numeric_columns]

    X_flat = [' '.join(row) for row in text[text_column].values.astype('U')]

    try: # Load the tokenizer from the file
        with open('tfidf_vectorizer.pkl', 'rb') as file:
            tfidf_vectorizer = pickle.load(file)
    except:
        print('create tokenizer_for_NLP first')

    X_tfidf = tfidf_vectorizer.transform(X_flat)
    X_text = pd.DataFrame(X_tfidf.toarray())
    X_text.columns = X_text.columns.astype(str)

    X_combined = pd.concat([X_text, numeric.reset_index(drop=True)], axis=1)

    for n, column in enumerate(y_columns):
        try:
            pretrained_model = load_model(name=f'catboost_classif{n}')
        except:
            print('no model in MLflow URL trying to find it localy')
        print('********predicting*******')
        y_pred = pretrained_model.predict(X_combined)
        your_dataset[new_columns_names[n]] = y_pred
        print('********done*******')

    if save_local_csv == True:
        your_dataset.to_csv('./raw_data_slim/result_CatBoost.csv', index=True)



def new_column_NLP(df_preprocessed):
    """
    adding new columns to df
    """
    ####################################

    for n, column in enumerate(y_columns):
        try:
            pretrained_model = load_model(name=f'catboost_classif{column[21:]}')
        except:
            print('no model in MLflow URL trying to find it localy')
        print('********predicting*******')
        y_pred = pretrained_model.predict()
        df_preprocessed[column] = y_pred
        print('********done*******')

    return df_preprocessed


def in_production_catboost(y_columns=y_columns):
    for n, column in enumerate(y_columns):
        mlflow_transition_model(f'catboost_classif{n}', 'None', 'Production')
    return print('model set as in production')
