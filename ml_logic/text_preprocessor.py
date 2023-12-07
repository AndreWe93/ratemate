"""
Text Preprocessor
"""
import pandas as pd
import numpy as np

import string

#!pip install nltk # Install NLTK
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def create_all_reviews_column(dataset):
    '''
    Function to create a new column with all the english reviews
    '''
    dataset['review_text_processed'] = dataset['textTranslated']
    if pd.isna(dataset['textTranslated']):
        dataset['review_text_processed'] = dataset['text']
    return dataset['review_text_processed']

class TextPreprocessor:

    def __init__(self, dataset):
        self.google_reviews_df = dataset

    def all_the_reviews(self):
        '''
        Create a new column with all the english reviews in the string format
        '''
        self.google_reviews_df['review_english'] = self.google_reviews_df.apply(create_all_reviews_column, axis=1)
        self.google_reviews_df['review_english'] = self.google_reviews_df['review_english'].astype(str)
        return self.google_reviews_df

    def preprocessing_with_stopwords(self, sentence):
        '''
        NLP preprocessing with stopwords (Stopwords are not removed)
        '''
        # Drop the nan values
        sentence = sentence.strip()
        sentence = sentence.lower()
        sentence = ''.join(char for char in sentence if not char.isdigit())

        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, '')

        tokenized_sentence = word_tokenize(sentence)

        lemmatized = [WordNetLemmatizer().lemmatize(word, pos="v") for word in tokenized_sentence]

        cleaned_sentence = ' '.join(word for word in lemmatized)

        if cleaned_sentence == 'nan' or cleaned_sentence == '' or cleaned_sentence == 'none':
            cleaned_sentence = np.nan

        return cleaned_sentence

    def preprocessing_without_stopwords(self, sentence):
        '''
        NLP preprocessing without stopwords (Stopwords are removed)
        '''
        # Drop the nan values
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

        if cleaned_sentence == 'nan' or cleaned_sentence == '' or cleaned_sentence == 'none':
            cleaned_sentence = np.nan

        return cleaned_sentence

    def preprocess_dataset(self):
        '''
        Preprocesses the dataset by executing all the above functions
        '''
        # Create column with all the english reviews
        self.google_reviews_df = self.all_the_reviews()

        # NLP preprocessing without stopwords removal
        self.google_reviews_df["reviews_with_SW"] = self.google_reviews_df['review_english'].apply(
            self.preprocessing_with_stopwords)
        self.google_reviews_df = self.google_reviews_df.dropna(subset=['reviews_with_SW'])

        # NLP preprocessing with stopwords removal
        self.google_reviews_df["reviews_without_SW"] = self.google_reviews_df['review_english'].apply(
            self.preprocessing_without_stopwords)
        self.google_reviews_df = self.google_reviews_df.dropna(subset=['reviews_without_SW'])

        # Dropping original reviews columns
        self.google_reviews_df.drop(columns=['text', 'textTranslated', 'review_english'], inplace=True)


        # # Converting publishedAtDate column format to datetime
        # self.google_reviews_df.publishedAtDate = pd.to_datetime(self.google_reviews_df.publishedAtDate)

if __name__ == "__main__":
    your_dataset = pd.read_csv("raw_data_slim/merged_slim_file.csv")
    text_preprocessor = TextPreprocessor(your_dataset)
    text_preprocessor.preprocess_dataset()
    print("Done")
    print(text_preprocessor.google_reviews_df.head(3))
    pass
