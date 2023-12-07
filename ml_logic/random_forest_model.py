from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml_logic.text_preprocessor import TextPreprocessor
from ml_logic.registry import mlflow_transition_model

import pandas as pd
import mlflow
from params import *
import pickle

from mlflow.models import infer_signature

# Columns definition
columns = [
        "reviewDetailedRating/Atmosphere",
        "reviewDetailedRating/Food",
        "reviewDetailedRating/Service",
        'reviews_without_SW',
        'reviews_with_SW',
        'stars'
        ]

y_columns = [
        "reviewDetailedRating/Atmosphere",
        "reviewDetailedRating/Food",
        "reviewDetailedRating/Service"
        ]

X_column = ['reviews_without_SW']

numeric_columns = ['stars']

new_columns_names = [
                     "atmosphere_rating",
                     "food_rating",
                     "service_rating"
                     ]


# Loading and Preprocessing the dataset
def load_dataset_train(file_path="./raw_data_slim/merged_slim_file.csv"):
    return pd.read_csv(file_path)

def preprocess_reviews_text_train(df):
    '''
    1. Creates new column with all the english reviews (Combination of #Text and #TranslatedText)
    2. NLP Preprocessing
    2.1 Creates a new column with NLP preprocessed text with Stopwords
    2.2 Creates a new column with NLP preprocessed text without Stopwords
    3. Keeps all the review columns except for the new NLP preprocessed text
    '''
    text_preprocessor = TextPreprocessor(df)
    text_preprocessor.preprocess_dataset()
    processed_df = text_preprocessor.google_reviews_df
    return processed_df

def tfidf_vectorizer(text):
    """
    Returns TF-IDF representation of text and saves the tokenizer to a 'pkl' file
    """
    # Convert text to TF-IDF representation
    X_flat = [' '.join(row) for row in text[X_column].values.astype('U')]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer = tfidf_vectorizer.fit(X_flat)
    X_tfidf = tfidf_vectorizer.transform(X_flat)

    # Save the tokenizer to a file
    with open('tfidf_vectorizer.pkl', 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)

    return X_tfidf


def basic_data_prep(your_dataset):
    '''
    Does basic data handling and returns X, y
    '''
    print("Preparing dataset...")
    data = your_dataset[columns].copy()
    data.dropna(inplace=True)

    y = data[y_columns]
    text = data[X_column]
    numeric = data[numeric_columns]

    print("X, y prepared ✅")

    return y, text, numeric


def basic_data_prep_predict(your_dataset):
    '''
    Does basic data handling and returns X, y
    '''
    print("Preparing dataset...")

    y = your_dataset[y_columns]
    text = your_dataset[X_column]
    numeric = your_dataset[numeric_columns]

    print("X, y prepared ✅")

    return y, text, numeric


def tfidf_vectorizer_train_data(your_dataset):
    '''
    fits with train data and returns tfidf vectorizer
    '''
    y = basic_data_prep(your_dataset)[0]
    text = basic_data_prep(your_dataset)[1]
    numeric = basic_data_prep(your_dataset)[2]

    X_tfidf = tfidf_vectorizer(text)

    X = pd.concat([pd.DataFrame(X_tfidf.toarray()), numeric.reset_index(drop=True)], axis=1)

    return X, y


def tfidf_vectorizer_predict_data(your_dataset):
    '''
    Loads the vectorizer and returns X
    '''
    y = basic_data_prep_predict(your_dataset)[0]
    text = basic_data_prep_predict(your_dataset)[1]
    numeric = basic_data_prep_predict(your_dataset)[2]

    X_flat = [' '.join(row) for row in text[X_column].values.astype('U')]

    try: # Load the tokenizer from the file
        with open('tfidf_vectorizer.pkl', 'rb') as file:
            tfidf_vectorizer = pickle.load(file)
    except:
        print('create tokenizer pkl file first')

    X_tfidf = tfidf_vectorizer.transform(X_flat)

    X = pd.concat([pd.DataFrame(X_tfidf.toarray()), numeric.reset_index(drop=True)], axis=1)

    return X, y


def train_model_random_forest(your_dataset):
    '''
    Trains the model and saves it to mlflow
    '''
    X = tfidf_vectorizer_train_data(your_dataset)[0]
    y = tfidf_vectorizer_train_data(your_dataset)[1]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name='ratemate_random_forest_multi')

    with mlflow.start_run() as run:
        n_estimators = 10
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        classifier = RandomForestClassifier(n_estimators=n_estimators)
        multi_target_classifier = MultiOutputClassifier(classifier)
        multi_target_classifier.fit(X_train, y_train)

        # Log main classifier
        signature = infer_signature(X_train, multi_target_classifier.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=multi_target_classifier,
            artifact_path="model",
            registered_model_name="ran_forest",
            signature=signature
        )

        params = multi_target_classifier.get_params()

        # Make predictions
        predictions = multi_target_classifier.predict(X_test)
        mlflow.log_params(params)

        # Compute and print metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        #
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)

        # params' default values are saved with ModelSignature
        signature = infer_signature(X_train, multi_target_classifier.predict(X_train))


        # Log models and metrics for each estimator
        for i, estimator in enumerate(multi_target_classifier.estimators_):
            estimator.fit(X_train, y_train.iloc[:, i])
            predictions_single = estimator.predict(X_test)
            mae_single = mean_absolute_error(y_test.iloc[:, i], predictions_single)
            mlflow.log_metric(f"MAE_{i}", mae_single)
            signature = infer_signature(X_train, multi_target_classifier.predict(X_train))

            # Log each model
            signature_single = infer_signature(X_train, estimator.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=estimator,
                artifact_path=f"model_{i}",
                registered_model_name=f"ran_forest_{i}",
                signature=signature_single
            )

    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    mlflow.end_run()


    print("✅ train() done \n")


def in_production(y_columns=y_columns):
    for n, column in enumerate(y_columns):
        mlflow_transition_model(f'random_forest_{n}', 'None', 'Production')
    return print('model set as in production')


def pred_from_random_forest(your_dataset):
    '''
    Predicts with the model
    '''
    X = tfidf_vectorizer_predict_data(your_dataset)[0]

    X.columns = X.columns.astype(str)

    #logged_model = 'runs:/{MLFLOW_RUN_ID}/model'
    logged_model = 'runs:/bd7884accc114cf2ae05e29a1e6da74f/model'
    loaded_model = mlflow.sklearn.load_model(logged_model)

    y_pred = loaded_model.predict(X)

    result_df = your_dataset
    result_df[new_columns_names] = y_pred

    print("\n✅ prediction done: ", new_columns_names)
    return result_df

if __name__ == "__main__":
    train_dataset = load_dataset_train()
    processed_df_train = preprocess_reviews_text_train(train_dataset)
    train_model_random_forest(processed_df_train)
