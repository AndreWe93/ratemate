import pandas as pd
# !pip install torch
# !pip install transformers

from transformers import pipeline, BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

'''Function to classify reviews based on pre-selected topics
    with pre-selected keywords. With the Bert model, the function will
    calculate similiarity between the keywords and the words in the review
    and classify the reivew accordingly
'''


def load_bert_model(model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return tokenizer, model

def define_topic_related_words():
    topic_related_words = {
        'price': ['price'],
        'service': ['service', 'staff'],
        'atmosphere': ['atmosphere', 'ambiance', 'decor', 'vibe', 'place', 'room', 'cosy', 'design', 'comfortable'],
        'food': ['food', 'dish', 'menu', 'flavor', 'quality', 'warm', 'cold', 'portion']
    }
    return topic_related_words

# Function to calculate normalized scores for a given review
def calculate_scores(review_text, tokenizer, model, topic_related_words):
    tokens = tokenizer(review_text, return_tensors='pt', truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs['last_hidden_state'].squeeze()

    similarity_scores = {}
    for topic, keywords in topic_related_words.items():
        topic_embeddings = [model.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids(keyword)].detach().numpy() for keyword in keywords]
        similarity_scores[topic] = cosine_similarity(embeddings.detach().numpy(), topic_embeddings).mean(axis=1).sum()

    softmax_scores = {topic: np.exp(score) / np.sum(np.exp(list(similarity_scores.values()))) for topic, score in similarity_scores.items()}
    rounded_scores = {topic: round(score, 2) for topic, score in softmax_scores.items()}

    return rounded_scores

def process_reviews(df, column_name): #column_name of review text to be analyzed
    tokenizer, model = load_bert_model()
    topic_related_words = define_topic_related_words()

    # Apply the scoring function to each review in the DataFrame
    df['topic_scores'] = df[f'{column_name}'].apply(lambda text: calculate_scores(text, tokenizer, model, topic_related_words))

    # Expand the scores into separate columns
    df = pd.concat([df.drop(['topic_scores'], axis=1), df['topic_scores'].apply(pd.Series)], axis=1)

    return df

# test to run a df of 50 reviews

if __name__ == "__main__":
    print("Loading data...")
    data = pd.read_csv("raw_data_slim/merged_slim_file.csv")
    data = data.dropna(subset=['textTranslated'])
    print("Data loaded ✅")
    df = data.head(50)
    result_df = process_reviews(df, "textTranslated")
    print(result_df.head(10))
