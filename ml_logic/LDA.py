from nltk.stem import RSLPStemmer
#from nltk.tokenize import word_tokenize
#import string
#import text_unidecode as unidecode
import pandas as pd
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from RateMate.ml_logic.text_preprocessor import TextPreprocessor


import nltk
#nltk.download("rslp")

stemmer = RSLPStemmer()


class LDA_Analysis:
    def __init__(self, dataset, max_df = 0.75,
                 max_features = 1000, ngram_range=(3,6),
                 n_components = 10, topwords = 15):
        self.dataset = dataset
        self.clean_reviews = dataset['reviews_without_SW']
        self.clean_reviews.dropna(inplace=True)
        self.max_df = max_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(max_df = self.max_df,
                                          max_features = self.max_features,
                                          ngram_range=self.ngram_range)
        self.model = self.LDA_model()
        self.topwords = topwords



    def vectorisation(self):
        if self.clean_reviews is None or len(self.clean_reviews) == 0:
            print("Error: No reviews after cleaning.")
            return None
        vectorized_reviews = pd.DataFrame(self.vectorizer.fit_transform(self.clean_reviews).toarray(),
                                        columns = self.vectorizer.get_feature_names_out())

        #print(f" vectorized_reviews.shape = {vectorized_reviews.shape}")

        return vectorized_reviews


    def LDA_model(self):
        vectorized_reviews = self.vectorisation()
        lda = LatentDirichletAllocation(n_components = self.n_components)
        model = lda.fit(vectorized_reviews)
        return model


    def document_mixture(self):
        document_mixture = self.model.transform(self.vectorisation())
        result = round(pd.DataFrame(document_mixture,
                        columns = [f"Topic {i+1}" for i in range(self.n_components)]).set_index(self.clean_reviews.index)
            ,2)
        print(document_mixture.shape)
        return result




    def topic_word(self, topic, with_weights = True):
        topwords_indexes = topic.argsort()[:-self.topwords - 1:-1]
        if with_weights == True:
            topwords = [(self.vectorizer.get_feature_names_out()[i], round(topic[i],2)) for i in topwords_indexes]
        if with_weights == False:
            topwords = [self.vectorizer.get_feature_names_out()[i] for i in topwords_indexes]
        return topwords


    def print_topics(self ):
        for idx, topic in enumerate(self.model.components_):
            print("-"*20)
            print("Topic %d:" % (idx))
            print(self.topic_word(topic))












if __name__ == "__main__":

    path = "./RateMate/raw_data/merged_slim_file.csv"

    text_preprocessor = TextPreprocessor(path)
    text_preprocessor.preprocess_dataset()

    df = text_preprocessor.google_reviews_df
    lda = LDA_Analysis(df)

    print("Done")
    print(lda.document_mixture().head())

    lda.print_topics()
