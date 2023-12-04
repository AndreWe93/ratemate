from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ml_logic.tfdf import numeric_columns,new_columns_names, text_column, y_columns, get_dataset_only_for_tfdf
import pandas as pd
import mlflow
from params import *

# Take user input for the file path or take the default path
default_file_path = "./raw_data_slim/merged_slim_file.csv"
#file_path = input(f"Enter full dataset for training with path, default value = '{default_file_path}'") or default_file_path

# Load the dataset
print("Loading dataset...")
your_dataset = pd.read_csv(default_file_path)
print("Data loaded ✅")

print("Preparing dataset...")
data = get_dataset_only_for_tfdf(your_dataset)
print("Dataset prepared ✅")

y = data[y_columns]
text = data[text_column]
numeric = data[numeric_columns]

print("X, y prepared ✅")

# Convert text to TF-IDF representation
X_flat = [' '.join(row) for row in text[text_column].values.astype('U')]
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_flat)
X_tfidf = tfidf_vectorizer.transform(X_flat)


# Combine text and numeric data
X = pd.concat([pd.DataFrame(X_tfidf.toarray()), numeric.reset_index(drop=True)], axis=1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training and saving models..")

# Set up MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(experiment_name='ratemate_random_forest_multi')

with mlflow.start_run() as run:
    n_estimators = 50
    # params = {'n_estimators': n_estimators}
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    classifier = RandomForestClassifier(n_estimators=n_estimators)
    multi_target_classifier = MultiOutputClassifier(classifier)
    multi_target_classifier.fit(X_train, y_train)
    params = multi_target_classifier.get_params()
    # Make predictions
    predictions = multi_target_classifier.predict(X_test)
    mlflow.log_params(params)

    # Compute and print metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)

    # Log models and metrics for each estimator
    for i, estimator in enumerate(multi_target_classifier.estimators_):
        estimator.fit(X_train, y_train.iloc[:, i])
        predictions_single = estimator.predict(X_test)
        mae_single = mean_absolute_error(y_test.iloc[:, i], predictions_single)
        mlflow.log_metric(f"MAE_{i}", mae_single)

        # Log each model
        mlflow.sklearn.log_model(
            sk_model=estimator,
            artifact_path=f"model_{i}",
            registered_model_name="ran_forest"
        )

"""
adding new columns to df
"""
####################################
#
####################################


text = your_dataset[text_column]
numeric = your_dataset[numeric_columns]
####################################
#
#####################
X_flat = [' '.join(row) for row in text[text_column].values.astype('U')]

X_tfidf = tfidf_vectorizer.transform(X_flat)
X_text = pd.DataFrame(X_tfidf.toarray())
X_text.columns = X_text.columns.astype(str)

X_combined = pd.concat([X_text, numeric.reset_index(drop=True)], axis=1)

y_pred = multi_target_classifier.predict(X_combined)
your_dataset[new_columns_names] = y_pred
print('********done*******')
your_dataset.to_csv('./raw_data_slim/result.csv', index=True)



# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'estimator__n_estimators': [50, 100, 200],
#     'estimator__max_depth': [None, 5, 10],
#     'estimator__min_samples_split': [2, 5, 10],
# }

# grid_search = GridSearchCV(multi_target_classifier, param_grid, cv=5, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# best_params = grid_search.best_params_
# print(best_params)

# {'estimator__max_depth': None,
#  'estimator__min_samples_split': 2,
#  'estimator__n_estimators': 100}