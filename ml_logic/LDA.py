from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import text_unidecode as unidecode

import nltk
nltk.download("rslp")

stemmer = RSLPStemmer()

stop_words = set(stopwords.words('english')) ## defining stopwords


def cleaning(sentence):

    # Basic cleaning
    sentence = sentence.lower() ## lowercasing
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## removing numbers
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## removing punctuation
    # Advanced cleaning
    tokenized_sentence = word_tokenize(sentence) ## tokenizing

    tokenized_sentence = [w for w in tokenized_sentence
                                  if not w in stop_words] ## remove stopwords
    stemmed_sentence = [stemmer.stem(word)
              for word in tokenized_sentence] ## get word stems
    decoded_sentence = [unidecode.unidecode(w) for w in stemmed_sentence] ## remove accents

    cleaned_sentence = ' '.join(decoded_sentence) ## join back into a string

    return cleaned_sentence


def topic_word(vectorizer, model, topic, topwords, with_weights = True):
    topwords_indexes = topic.argsort()[:-topwords - 1:-1]
    if with_weights == True:
        topwords = [(vectorizer.get_feature_names_out()[i], round(topic[i],2)) for i in topwords_indexes]
    if with_weights == False:
        topwords = [vectorizer.get_feature_names_out()[i] for i in topwords_indexes]
    return topwords


def print_topics(vectorizer, model, topwords):
    for idx, topic in enumerate(model.components_):
        print("-"*20)
        print("Topic %d:" % (idx))
        print(topic_word(vectorizer, model, topic, topwords))
