import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models, Model, utils
from ml_logic.text_preprocessor import TextPreprocessor
from ml_logic.registry import *
from params import *
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

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])

    return model, X_pad




def fit_NLP(model, X_pad, y, n=0): #used within 'pretrained_NLP_models'

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


# def evaluate_NLP(model,X_pad_test,y_test
#                  ):

#     """
#     """
#     y_pred = model.predict(X_pad_test)

#     loss, mae = model.evaluate(X_pad_test,y_test)



#     #return X_test, y_test[:,n], y_pred, loss, mae
#     return loss, mae


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

        if pretrained_model is None:
            print("Model loading failed.")

        else:
            print(f'********predicting*******\n')
            y_pred = predict_NLP(pretrained_model, X)
            df_preprocessed[new_columns_names[n]] = y_pred
            print('********done*******')



    return df_preprocessed


def in_production():
    for n, column in enumerate(y_columns):
        mlflow_transition_model(f'CNN_{column[21:]}', 'None', 'Production')
    return print('models set as in production')


if __name__ == "__main__":
    path = "./raw_data_slim/Pepenero Schwabing.csv"
    dataset = pd.read_csv(path)

def build_model_nlp(X, maxlen = maxlen, embedding_size = embedding_size):

    tk = tokenizer_for_NLP(X)
    X_tokens = tk.texts_to_sequences(X.tolist())
    X_pad = pad_sequences(X_tokens, dtype=float, padding='post', maxlen=maxlen)

    vocab_size = len(tk.word_index)

    input_layer = layers.Input(shape=(maxlen,), dtype='int32')

    embedding = layers.Embedding(input_dim=vocab_size + 1, input_length=maxlen, output_dim=embedding_size, mask_zero=True)(input_layer)

    conv1d = layers.Conv1D(128, 5, padding='same', activation='relu')(embedding)
    maxpool = layers.GlobalMaxPooling1D()(conv1d)

    dense = layers.Dense(64, activation='relu')(maxpool)
    dropout = layers.Dropout(0.1)(dense)
    dense_2 = layers.Dense(32, activation='relu')(dropout)

    # Three output neurons for each rating
    atmosphere_output = layers.Dense(1, activation='linear', name='atmosphere_output')(dense_2)
    food_output = layers.Dense(1, activation='linear', name='food_output')(dense_2)
    service_output = layers.Dense(1, activation='linear', name='service_output')(dense_2)

    model = models.Model(inputs=input_layer, outputs=[atmosphere_output, food_output, service_output])

    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])

#     model = Sequential([
#         layers.Embedding(input_dim=vocab_size+1, input_length=maxlen, output_dim=embedding_size, mask_zero=True),
#         layers.Conv1D(10, kernel_size=15, padding='same', activation="relu"),
#         layers.Conv1D(10, kernel_size=10, padding='same', activation="relu"),
#         layers.Flatten(),
#         layers.Dense(30, activation='relu'),
#         layers.Dropout(0.15),
#         layers.Dense(1, activation='relu'),
#     ])

#     model.compile(loss="mse", optimizer=Adam(learning_rate=1e-4), metrics=['mae'])
    return model, X_pad


def build_model_num(X_num, y):
    input_num = layers.Input(shape=(X_num.shape[1],))

    x = layers.Dense(64, activation="relu")(input_num)
    x = layers.Dense(32, activation="relu")(x)
    output_num = layers.Dense(1, activation="relu")(x)

    model_num = models.Model(inputs=input_num, outputs=output_num)

    model_num.compile(loss = "mse", optimizer=Adam(learning_rate=5e-4), metrics=['mae'])
    model_num.fit(X_num, y,
          validation_split=0.3,
          epochs=50,
          batch_size=32,
          callbacks=[es]
          )

    return model_num


def combined_models():

    model_nlp = build_model_nlp() # comment-out to keep pre-trained weights not to start from scratch
    input_text = model_nlp.input
    output_text = model_nlp.output

    model_num = build_model_num() # comment-out to keep pre-trained weights not to start from scratch
    input_num = model_num.input
    output_num = model_num.output

    inputs = [input_text, input_num]

    combined = layers.concatenate([output_text, output_num])

    x = layers.Dense(10, activation="relu")(combined)

    outputs = layers.Dense(1, activation="linear")(x)

    model_combined = models.Model(inputs=inputs, outputs=outputs)

    utils.plot_model(model_combined, "multi_input_model.png", show_shapes=True)

    return model_combined


def compile_fit_combined_models(model_combined, X_pad, X_num):
    model_combined.compile(loss="mse", optimizer=Adam(learning_rate=1e-4), metrics=['mae'])
    es = EarlyStopping(patience=2)

    history = model_combined.fit(x=[X_pad, X_num],
                    y=y,
                    validation_split=0.3,
                    epochs=100,
                    batch_size=32,
                    callbacks=[es])
    return history
