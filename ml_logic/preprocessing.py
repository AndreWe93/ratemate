"""
Preprocessing functions
"""
import pandas as pd
import numpy as np

import string

#!pip install nltk # Install NLTK
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def create_review_text(dataset):
    dataset['review_text_processed'] = dataset['textTranslated']
    if pd.isna(dataset['textTranslated']):
        dataset['review_text_processed'] = dataset['text']
    return dataset['review_text_processed']

def dataset_intial_import():
    """
    Import dataset and add review_english column
    """
    google_reviews_df = pd.read_csv("raw_data/merged_slim_file.csv")
    google_reviews_df['review_english'] = google_reviews_df.apply(create_review_text, axis=1)
    google_reviews_df['review_english'] = google_reviews_df['review_english'].astype(str)
    return google_reviews_df


def preprocessing_with_stopwords(sentence):
    '''
    Basic preprocessing of the review text
    Stopwords are not removed
    '''
    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercase
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## remove numbers

    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## remove punctuation

    tokenized_sentence = word_tokenize(sentence) ## tokenize

    lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "v")
        for word in tokenized_sentence
    ]

    cleaned_sentence = ' '.join(word for word in lemmatized)

    if cleaned_sentence == 'nan':
        cleaned_sentence = np.nan

    return cleaned_sentence


def preprocessing_without_stopwords(sentence):
    '''
    Basic preprocessing of the review text
    Stopwords are not removed
    '''
    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercase
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## remove numbers

    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## remove punctuation

    tokenized_sentence = word_tokenize(sentence) ## tokenize
    stop_words = set(stopwords.words('english')) ## define stopwords

    tokenized_sentence_cleaned = [ ## remove stopwords
        w for w in tokenized_sentence if not w in stop_words
    ]

    lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "v")
        for word in tokenized_sentence_cleaned
    ]

    cleaned_sentence = ' '.join(word for word in lemmatized)
    if cleaned_sentence == 'nan':
        cleaned_sentence = np.nan

    return cleaned_sentence

if __name__ == "__main__":
    # Import the dataset and intial handling
    google_reviews_df = dataset_intial_import()

    # Dataset after basic preprocessing with stopwords
    google_reviews_df["reviews_with_SW"] = google_reviews_df['review_english'].apply(preprocessing_with_stopwords)

    # Dataset after basic preprocessing with stopwords
    google_reviews_df["reviews_without_SW"] = google_reviews_df['review_english'].apply(preprocessing_without_stopwords)

    print("Done")
    print(google_reviews_df.head(3))
