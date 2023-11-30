import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from RateMate.ml_logic.text_preprocessor import TextPreprocessor




import seaborn as sns

"""
constants
"""

path = "Ratemate/raw_data/merged_slim_file.csv"
dataset = pd.read_csv(path)


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

maxlen = 150

es = EarlyStopping(patience=5)
validation_split=0.3
epochs=50
batch_size=16
embedding_size = 150



def get_dataset_for_NLP(dataset=dataset):
    """
    returns preprocessed: DF for training NLP and original DF
    """
    text_preprocessor = TextPreprocessor(dataset)
    text_preprocessor.preprocess_dataset()

    df_full = text_preprocessor.google_reviews_df


    data = df_full[columns]
    data.dropna(inplace=True)

    return data, df_full



def get_Xy_for_NLP(data):
    """
    return X,y for df
    """
    y = data[y_columns].values
    X = data[X_column].values
    return X, y



def tokenizer_for_NLP(X):
    """
    returned tokenizer traind by X
    """
    X_words = [text_to_word_sequence(str(sentence)) for sentence in X]

    tk = Tokenizer()
    tk.fit_on_texts(X_words)
    return tk



def build_model_nlp(X, maxlen = maxlen, embedding_size = embedding_size): #used within 'pretrained_NLP_models'
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
        layers.Conv1D(20, kernel_size=15, padding='same', activation="relu"),
        layers.Conv1D(10, kernel_size=10, padding='same', activation="relu"),
        layers.Flatten(),
        layers.Dense(30, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='linear'),
    ])

#     model.compile(loss='categorical_crossentropy', # different from binary_crossentropy because we have multiple classes
#                   optimizer='adam', metrics=['accuracy'])
#     return model

# model_nlp = build_model_nlp()

# def build_model_nlp():
#     model = Sequential([
#         layers.Embedding(input_dim=vocab_size+1, input_length=maxlen, output_dim=embedding_size, mask_zero=True),
#         layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
#         layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(1, activation='linear'),
#     ])

    model.compile(loss='mean_squared_error', # Using mean squared error for regression
                  optimizer=Adam(), metrics=['mae'])  # Mean Absolute Error as a metric
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



def evaluate_NLP(model, tk,
                 df_test_csv_path = 'RateMate/raw_data/Pepenero Schwabing.csv',
                 maxlen = maxlen,
                 n = 0):

    """
    """
    df_test = get_dataset_for_NLP(df_test_csv_path)
    X_test, y_test = get_Xy_for_NLP(df_test)

    X_words_test = [text_to_word_sequence(str(sentence)) for sentence in X_test]
    X_tokens_test = tk.texts_to_sequences(X_words_test)

    X_pad_test = pad_sequences(X_tokens_test, dtype=float, padding='post', maxlen=maxlen)
    y_pred = model.predict(X_pad_test)

    loss, mae = model.evaluate(X_pad_test,y_test[:,n])



    return X_test, y_test[:,n], y_pred, loss, mae



def info_NLP(tk,X,y):
    """
    just info about X, y
    """
    X_tokens = tk.texts_to_sequences(X.tolist())

    sns.histplot([len(x) for x in X_tokens]);

    print('Imbalanced dataset:',pd.DataFrame(np.unique(y, return_counts=True)))



def predict_NLP(model, X, tk): #used within 'new_column_NLP'
    """
    tokenizing and padding X and making predictions
    """
    X_words_test = [text_to_word_sequence(str(sentence)) for sentence in X]
    X_tokens_test = tk.texts_to_sequences(X_words_test)

    X_pad_test = pad_sequences(X_tokens_test, dtype=float, padding='post', maxlen=maxlen)
    y_pred = model.predict(X_pad_test)

    return y_pred



def pretrained_NLP_models(X,y):
    """
    saving models for loading in 'new_column_NLP'
    """
    for n, column in enumerate(y_columns):
        model, X_pad = build_model_nlp(X)
        history = fit_NLP(model, X_pad, y, n=n)
        models.save_model(model, f'my_model_{n}')



def new_column_NLP(df_preprocessed, tk):
    """
    adding new columns to df
    """

    ####################################
    #
    ####################################

    # text_preprocessor = TextPreprocessor(path)
    # text_preprocessor.preprocess_dataset()
    # df = text_preprocessor.google_reviews_df
    X, y = get_Xy_for_NLP(df_preprocessed)
    ####################################
    #
    ####################################
    for n, column in enumerate(y_columns):

        pretrained_model = models.load_model(f'my_model_{n}')
        y_pred = predict_NLP(pretrained_model, X, tk)
        df_preprocessed[column[21:]] = y_pred

    return df_preprocessed





if __name__ == "__main__":
    pass
