import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from ml_logic.registry import *
from params import *
import pickle
import seaborn as sns
import mlflow.pyfunc


"""
constants
"""

maxlen = 200
es = EarlyStopping(patience=2)
validation_split=0.3
epochs=25
batch_size=32
embedding_size = 200
learning_rate = 0.01



def get_dataset_for_NLP(dataset):

    """
    returns preprocessed: DF for training NLP
    """

    # text_preprocessor = TextPreprocessor(dataset)
    # text_preprocessor.preprocess_dataset()

    # df_full = text_preprocessor.google_reviews_df
    #df_full.to_csv('./raw_data_slim/merged_slim_file.csv', index=True)
    data = dataset[columns].copy()
    data.dropna(inplace=True)

    return data

def get_Xy_for_NLP(data):
    """
    returns X,y for df
    """
    y = data[y_columns].values
    X = data[X_column].values
    X_num = data[numeric_columns].values

    return X, y, X_num


def tokenizer_for_NLP(X): #used in build_model_nlp
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


def fit_NLP(model, X_pad, y): #used within 'pretrained_NLP_models'

    """
    """

    history = model.fit(X_pad, y,
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
    X_test, y_test, X_num_test = get_Xy_for_NLP(df_test)

    X_words_test = [text_to_word_sequence(str(sentence)) for sentence in X_test]
    X_tokens_test = tk.texts_to_sequences(X_words_test)

    X_pad_test = pad_sequences(X_tokens_test, dtype=float, padding='post', maxlen=maxlen)

    return X_pad_test, y_test, X_num_test


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

def build_model_nlp(X, maxlen = maxlen, embedding_size =embedding_size):
    '''
    used in combined_models_NLP
    '''

    tk = tokenizer_for_NLP(X)
    #X = [str(item) for item in X]
    X_tokens = tk.texts_to_sequences(X.tolist())
    X_pad = pad_sequences(X_tokens, dtype=float, padding='post', maxlen=maxlen)

    vocab_size = len(tk.word_index)

    input_layer = layers.Input(shape=(maxlen,), dtype='int32')

    embedding = layers.Embedding(input_dim=vocab_size + 1,
                                 input_length=maxlen,
                                 output_dim=embedding_size,
                                 mask_zero=True)(input_layer)

    conv1d = layers.Conv1D(512, 15, padding='same', activation='relu')(embedding)

    # maxpool = layers.GlobalMaxPooling1D()(conv1d)
    flatt = layers.Flatten()(conv1d)
    dense = layers.Dense(256, activation='relu')(flatt)
    dropout = layers.Dropout(0.1)(dense)
    dense_2 = layers.Dense(128, activation='relu')(dropout)
    dropout = layers.Dropout(0.1)(dense_2)
    dense_2 = layers.Dense(64, activation='relu')(dropout)
    dropout = layers.Dropout(0.1)(dense_2)
    dense_2 = layers.Dense(32, activation='relu')(dropout)

    # Three output neurons for each rating
    atmosphere_output_text = layers.Dense(1, activation='linear', name='atmosphere_output_text')(dense_2)
    food_output_text = layers.Dense(1, activation='linear', name='food_output_text')(dense_2)
    service_output_text = layers.Dense(1, activation='linear', name='service_output_text')(dense_2)

    model = models.Model(inputs=input_layer, outputs=[atmosphere_output_text, food_output_text, service_output_text])

    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])
    return model, X_pad


def build_model_num(X_num):
    '''
    used in combined_models_NLP
    '''

    input_num = layers.Input(shape=(X_num.shape[1],))

    x = layers.Dense(6, activation="relu")(input_num)
    x = layers.Dense(3, activation="relu")(x)

    atmosphere_output_num = layers.Dense(1, activation='linear', name='atmosphere_output_num')(x)
    food_output_num  = layers.Dense(1, activation='linear', name='food_output_num')(x)
    service_output_num  = layers.Dense(1, activation='linear', name='service_output_num')(x)

    model_num = models.Model(inputs=input_num, outputs=[atmosphere_output_num, food_output_num, service_output_num])

    model_num.compile(loss = "mse", optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])

    return model_num


def combined_models_NLP(X, X_num):

    model_nlp, X_pad = build_model_nlp(X) # comment-out to keep pre-trained weights not to start from scratch
    input_text = model_nlp.input
    output_text = model_nlp.output

    model_num = build_model_num(X_num) # comment-out to keep pre-trained weights not to start from scratch
    input_num = model_num.input
    output_num = model_num.output



    inputs = [input_text, input_num]
    output_text_list = [output_text[0], output_text[1], output_text[2]]
    output_num_list = [output_num[0], output_num[1], output_num[2]]

    combined = layers.concatenate(output_text_list + output_num_list)

    x = layers.Dense(10, activation="relu")(combined)


    # Combine the outputs of both models into three final outputs
    atmosphere_output = layers.Dense(1, activation='linear', name='atmosphere_output')(x)
    food_output = layers.Dense(1, activation='linear', name='food_output')(x)
    service_output = layers.Dense(1, activation='linear', name='service_output')(x)


    model_combined = models.Model(inputs=inputs, outputs=[
        atmosphere_output, food_output, service_output])



    return model_combined, X_pad


def compile_fit_combined_models(model_combined, X_pad, X_num,y):

    model_combined.compile(loss="mse", optimizer=Adam(learning_rate=1e-4),
                           metrics=['mae'])

    history = model_combined.fit(x=[X_pad, X_num],
                    y=y,
                    validation_split=validation_split,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[es])
    return history


#@mlflow_run
def training_NLP_combine_model_and_saving_to_mlflow(X,y, X_num):
    """
    saving models for loading in 'new_column_NLP'
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name='ratemate_NLP_CNN')
    with mlflow.start_run() as run:

        mlflow.tensorflow.autolog()

        model, X_pad = combined_models_NLP(X, X_num)
        compile_fit_combined_models(model, X_pad, X_num,y)


        # params = model.get_params()
        # mlflow.log_params(params)

        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=f'CNN_combined'

        )


        print("✅ Model saved to MLflow")


def predict_NLP(model, X, X_num): #used within 'new_column_NLP'
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
    y_pred = model.predict([X_pad_test,X_num])

    return y_pred


def new_column_NLP(df_preprocessed, save_local_csv=False):
    """
    adding new columns to df
    """

    ####################################
    #
    ####################################
    X, y, X_num = get_Xy_for_NLP(df_preprocessed)
    ####################################
    #
    ####################################

    pretrained_model = load_model(name='CNN_combined', model_type='TF')

    if pretrained_model is None:
        print("Model loading failed.")

    else:
        print(f'********predicting*******\n')
        y_pred = predict_NLP(pretrained_model, X, X_num)
        df_preprocessed[new_columns_names] = y_pred
        print('********done*******')

    if save_local_csv == True:

        df_preprocessed.to_csv('./raw_data_slim/NLP_CNN_result.csv', index=True)
        print('********RF_result.csv saved ✅ at /raw_data_slim/*******')


    return df_preprocessed


def in_production():
    mlflow_transition_model('CNN_combined', 'None', 'Production')
    return print('models set as in production')
