"""
Text Preprocessor
"""
import pandas as pd
import numpy as np

import string

#!pip install nltk # Install NLTK
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.google_reviews_df = self.dataset_initial_import()

    def create_review_text(self, dataset):
        dataset['review_text_processed'] = dataset['textTranslated']
        if pd.isna(dataset['textTranslated']):
            dataset['review_text_processed'] = dataset['text']
        return dataset['review_text_processed']

    def dataset_initial_import(self):
        google_reviews_df = pd.read_csv(self.dataset_path)
        google_reviews_df['review_english'] = google_reviews_df.apply(self.create_review_text, axis=1)
        google_reviews_df['review_english'] = google_reviews_df['review_english'].astype(str)
        return google_reviews_df

    def preprocessing_with_stopwords(self, sentence):
        sentence = sentence.strip()
        sentence = sentence.lower()
        sentence = ''.join(char for char in sentence if not char.isdigit())

        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, '')

        tokenized_sentence = word_tokenize(sentence)

        lemmatized = [WordNetLemmatizer().lemmatize(word, pos="v") for word in tokenized_sentence]

        cleaned_sentence = ' '.join(word for word in lemmatized)

        if cleaned_sentence == 'nan':
            cleaned_sentence = np.nan

        return cleaned_sentence

    def preprocessing_without_stopwords(self, sentence):
        sentence = sentence.strip()
        sentence = sentence.lower()
        sentence = ''.join(char for char in sentence if not char.isdigit())

        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, '')

        tokenized_sentence = word_tokenize(sentence)
        stop_words = set(stopwords.words('english'))

        tokenized_sentence_cleaned = [w for w in tokenized_sentence if not w in stop_words]

        lemmatized = [WordNetLemmatizer().lemmatize(word, pos="v") for word in tokenized_sentence_cleaned]

        cleaned_sentence = ' '.join(word for word in lemmatized)
        if cleaned_sentence == 'nan':
            cleaned_sentence = np.nan

        return cleaned_sentence

    def preprocess_dataset(self):
        self.google_reviews_df["reviews_with_SW"] = self.google_reviews_df['review_english'].apply(
            self.preprocessing_with_stopwords)
        self.google_reviews_df["reviews_without_SW"] = self.google_reviews_df['review_english'].apply(
            self.preprocessing_without_stopwords)
        self.google_reviews_df.drop(columns=['text', 'textTranslated', 'review_english'], inplace=True)
        self.google_reviews_df.publishedAtDate = pd.to_datetime(self.google_reviews_df.publishedAtDate)

if __name__ == "__main__":
    dataset_path = "raw_data_slim/merged_slim_file.csv"
    text_preprocessor = TextPreprocessor(dataset_path)
    text_preprocessor.preprocess_dataset()
    print("Done")
    print(text_preprocessor.google_reviews_df.head(3))
