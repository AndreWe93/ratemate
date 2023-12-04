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
        'stars'
        ]
y_columns = [#"reviewContext/Price per person",
        #"reviewContext/Service",
        "reviewDetailedRating/Atmosphere",
        "reviewDetailedRating/Food",
        "reviewDetailedRating/Service"
        ]
text_column = [ 'reviews_without_SW',
        #'reviews_with_SW'
        ]
numeric_columns = ['stars']

new_columns_names = [#"price_rating",
                     "atmosphere_rating",
                     "food_rating",
                     "service_rating"
                     ]



es = EarlyStopping(patience=2)
validation_split=0.3
epochs=15
batch_size=32
embedding_size = 200
learning_rate = 0.001



def get_dataset_for_tfdf(df_full):

    """
    returns preprocessed: DF for training NLP and original DF
    # """

    # text_preprocessor = TextPreprocessor(dataset)
    # text_preprocessor.preprocess_dataset()

    # df_full = text_preprocessor.google_reviews_df

    data = df_full[columns].copy()
    data.dropna(inplace=True)

    return data, df_full

def get_dataset_only_for_tfdf(dataset):
    """
    returns preprocessed: DF only for training NLP
    """


    data = dataset[columns].copy()
    data.dropna(inplace=True)

    return data



# def get_Xy_for_NLP(data):
#     """
#     returns X,y for df
#     """
#     y = data[y_columns].values
#     X = data[X_column].values
#     return X, y



# def new_columns_tfdf(df_preprocessed):
#     """
#     adding new columns to df
#     """

#     ####################################
#     #
#     ####################################
#     X, y = get_Xy_for_NLP(df_preprocessed)
#     ####################################
#     #
#     ####################################
#     for n, column in enumerate(y_columns):
#         try:
#             pretrained_model = load_model(name='tf_df')
#         except:
#             print('no model in MLflow URL trying to find it localy')
#         print('********predicting*******')
#         y_pred = predict_NLP(pretrained_model, X)
#         df_preprocessed[column] = y_pred
#         print('********done*******')

#     return df_preprocessed




# def in_production():
#     for n, column in enumerate(y_columns):
#         mlflow_transition_model(f'CNN_{column[21:]}', 'None', 'Production')
#     return print('models set as in production')




if __name__ == "__main__":
    path = "./raw_data_slim/Pepenero Schwabing.csv"
    dataset = pd.read_csv(path)
