import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from interface.main import *
from params import *


def generate_wordcloud(df):
    fig = plt.figure(figsize=(30,30))
    wc = WordCloud(max_words=10000, min_font_size=10, height=800, width=1600,
               background_color="white", colormap="viridis").generate(" ".join(df["reviews_without_SW"].astype(str)))
    return fig

def generate_ratings_distribution(df):
    # Display Rating distribution
    fig = plt.figure(figsize=(10,10))
    dist = sns.countplot(data=df, x="stars", palette="Spectral", hue="stars")
    return fig

if __name__ == "__main__":
    whole_dataset = load_dataset()
    small_dataset = whole_dataset.head(10).copy()
    print("Loaded dataset and created small dataset ✅")

    processed_df_small = preprocess_reviews_text(small_dataset)
    # processed_df = preprocess_reviews_text(your_dataset)
    print("Preprocessed dataset ✅")

    word_cloud = generate_wordcloud(processed_df_small)
    # classify_reviews_df = process_reviews(processed_df, "reviews_without_SW")
    print(word_cloud)
    pass
