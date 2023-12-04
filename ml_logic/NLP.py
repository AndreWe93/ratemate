import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras import models, Model
from ml_logic.text_preprocessor import TextPreprocessor
from ml_logic.registry import load_model, mlflow_transition_model, mlflow_run, save_results
from params import *
import json
import pickle


import seaborn as sns

#pip install mlflow
import mlflow.pyfunc
#from mlflow.keras import save_model


"""
constants
"""

#path = "raw_data/merged_slim_file.csv"
#dataset = pd.read_csv(path)


columns = [#"reviewContext/Price per person",
        #"reviewContext/Service",
        "reviewDetailedRating/Atmosphere",
        "reviewDetailedRating/Food",
        "reviewDetailedRating/Service",
        'reviews_without_SW',
        'reviews_with_SW',
        ]
y_columns = [#"reviewContext/Price per person",
        #"reviewContext/Service",
        "reviewDetailedRating/Atmosphere",
        "reviewDetailedRating/Food",
        "reviewDetailedRating/Service"
        ]
X_column = [ 'reviews_without_SW',
        #'reviews_with_SW'
        ]

new_columns_names = [#"price_rating",
                     "atmosphere_rating",
                     "food_rating",
                     "service_rating"
                     ]


maxlen = 200

es = EarlyStopping(patience=2)
validation_split=0.3
epochs=15
batch_size=32
embedding_size = 200
learning_rate = 0.001



def get_dataset_for_NLP(dataset):

    """
    returns preprocessed: DF for training NLP and original DF
    """

    # text_preprocessor = TextPreprocessor(dataset)
    # text_preprocessor.preprocess_dataset()

    # df_full = text_preprocessor.google_reviews_df
    #df_full.to_csv('./raw_data_slim/merged_slim_file.csv', index=True)
    data = dataset[columns].copy()
    data.dropna(inplace=True)

    return data

def get_dataset_only_for_NLP(dataset):
    """
    returns preprocessed: DF only for training NLP
    """


    data = dataset[columns].copy()
    data.dropna(inplace=True)

    return data



def get_Xy_for_NLP(data):
    """
    returns X,y for df
    """
    y = data[y_columns].values
    X = data[X_column].values
    return X, y



def tokenizer_for_NLP(X): #used in build_model_nlp_CNN
    """
    returned tokenizer traind by X
    """
    X_words = [text_to_word_sequence(str(sentence)) for sentence in X]

    tk = Tokenizer()
    tk.fit_on_texts(X_words)
    # Save the tokenizer to a file
    with open('tokenizer.pkl', 'wb') as file:
        pickle.dump(tk, file)
    return tk



def build_model_nlp_CNN(X, maxlen = maxlen, embedding_size = embedding_size): #used within 'pretrained_NLP_models'
    """
    buolding main model for NLP
    """
    # Vocab size
    tk = tokenizer_for_NLP(X)

    X_tokens = tk.texts_to_sequences(X.tolist())
    X_pad = pad_sequences(X_tokens, dtype=float, padding='post', maxlen=maxlen)

    vocab_size = len(tk.word_index)


    model = Sequential([
        layers.Embedding(input_dim=vocab_size+1, input_length=maxlen, output_dim=embedding_size, mask_zero=True),
        layers.Conv1D(128, 5, padding='same', activation='relu'),
        layers.GlobalMaxPooling1D(),
        #layers.Conv1D(30, kernel_size=15, padding='same', activation="relu"),
        #layers.Conv1D(20, kernel_size=10, padding='same', activation="relu"),
        #layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(32, activation='relu'),

        #layers.Dense(30, activation='relu'),
        layers.Dense(1, activation='linear'),

        #Dense(3)  # Три выходных нейрона для каждой оценки

                ])

    model.compile(loss='mean_squared_error', # Using mean squared error for regression
                  optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])  # Mean Absolute Error as a metric
    return model, X_pad




def fit_NLP(model, X_pad, y, maxlen = maxlen, n=0): #used within 'pretrained_NLP_models'

    """
    """

    history = model.fit(X_pad, y[:,n],
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es]
            )
    return history

def get_test_data_for_evaluate_NLP(
                 df_test_csv_path = './raw_data_slim/Pepenero Schwabing.csv',
                 maxlen = maxlen):

    """
    """
    try: # Load the tokenizer from the file
        with open('tokenizer.pkl', 'rb') as file:
            tk = pickle.load(file)

    except:
        print('create tokenizer_for_NLP first')

    df_test = get_dataset_for_NLP(pd.read_csv(df_test_csv_path))
    X_test, y_test = get_Xy_for_NLP(df_test)

    X_words_test = [text_to_word_sequence(str(sentence)) for sentence in X_test]
    X_tokens_test = tk.texts_to_sequences(X_words_test)

    X_pad_test = pad_sequences(X_tokens_test, dtype=float, padding='post', maxlen=maxlen)

    return X_pad_test, y_test



def evaluate_NLP(model,X_pad_test,y_test,
                 n = 0):

    """
    """
    y_pred = model.predict(X_pad_test)

    loss, mae = model.evaluate(X_pad_test,y_test[:,n])



    #return X_test, y_test[:,n], y_pred, loss, mae
    return loss, mae



def info_NLP(X,y):
    """
    just info about X, y

    """
    try: # Load the tokenizer from the file
        with open('tokenizer.pkl', 'rb') as file:
            tk = pickle.load(file)

    except:
        print('create tokenizer_for_NLP first')

    X_tokens = tk.texts_to_sequences(X.tolist())

    sns.histplot([len(x) for x in X_tokens]);

    print('Imbalanced dataset:',pd.DataFrame(np.unique(y, return_counts=True)))



#@mlflow_run
def pretraining_NLP_models(X,y):
    """
    saving models for loading in 'new_column_NLP'
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name='ratemate_NLP_CNN')
    for n, column in enumerate(y_columns):
         with mlflow.start_run() as run:

            mlflow.tensorflow.autolog()

            model, X_pad = build_model_nlp_CNN(X)
            history = fit_NLP(model, X_pad, y, n=n)

            mlflow.tensorflow.log_model(
                model=model,
                artifact_path="model",
                registered_model_name=f'CNN_{column[21:]}',

            )


            print("✅ Model saved to MLflow")


def predict_NLP(model, X): #used within 'new_column_NLP'
    """
    tokenizing and padding X and making predictions
    """
    try: # Load the tokenizer from the file
        with open('tokenizer.pkl', 'rb') as file:
            tk = pickle.load(file)

    except:
        print('create tokenizer_for_NLP first')


    X_words_test = [text_to_word_sequence(str(sentence)) for sentence in X]
    X_tokens_test = tk.texts_to_sequences(X_words_test)

    X_pad_test = pad_sequences(X_tokens_test, dtype=float, padding='post', maxlen=maxlen)
    y_pred = model.predict(X_pad_test)

    return y_pred



def new_column_NLP(df_preprocessed):
    """
    adding new columns to df
    """

    ####################################
    #
    ####################################
    X, y = get_Xy_for_NLP(df_preprocessed)
    ####################################
    #
    ####################################
    for n, column in enumerate(y_columns):
        pretrained_model = load_model(name=f'CNN_{column[21:]}')
        print('********predicting*******')
        y_pred = predict_NLP(pretrained_model, X)
        df_preprocessed[column] = y_pred
        print('********done*******')

    return df_preprocessed




def in_production():
    for n, column in enumerate(y_columns):
        mlflow_transition_model(f'CNN_{column[21:]}', 'None', 'Production')
    return print('models set as in production')




if __name__ == "__main__":
    path = "./raw_data_slim/Pepenero Schwabing.csv"
    dataset = pd.read_csv(path)
