# pip install apify_client
import pandas as pd
import numpy as np
from apify_client import ApifyClient

def scrape_apify(url, maxReviews = 200, reviewsSort = "newest", language = "en", personalData = True,
                 token = "apify_api_Nrud6TSWGlModOZyMcaSFyTpnY73hF1Kob5x"):
    # Initialize the ApifyClient with your API token
    client = ApifyClient(token) # AWs token, will expire 30.11.2023

    # Prepare the Actor input
    run_input = {
        "startUrls": [{ "url": f'{url}'}],
        "maxReviews": maxReviews,
        "reviewsSort": reviewsSort,
        "language": language,
        "personalData": personalData,
    }
    columns = [
        "placeId",
        "title",
        "reviewId",
        "reviewerId",
        "isLocalGuide",
        "reviewDetailedRating/Atmosphere",
        "reviewDetailedRating/Food",
        "reviewDetailedRating/Service",
        "reviewerNumberOfReviews",
        "text",
        "textTranslated",
        "stars"
    ]

    # Run the Actor and wait for it to finish
    run = client.actor("Xb8osYTtOjlsgI6k9").call(run_input=run_input)

    df = pd.DataFrame(columns=columns)

    for i, item in enumerate(client.dataset(run["defaultDatasetId"]).iterate_items()):
        df.loc[i, "placeId"] = item["placeId"]
        df.loc[i, "title"] = item["title"]
        df.loc[i, "reviewId"] = item["reviewId"]
        df.loc[i, "reviewerId"] = item["reviewerId"]
        df.loc[i, "isLocalGuide"] = item["isLocalGuide"]
        if "reviewDetailedRating" in item and "Food" in item["reviewDetailedRating"]:
            df.loc[i, "reviewDetailedRating/Food"] = item["reviewDetailedRating"]["Food"]
        else:
            df.loc[i, "reviewDetailedRating/Food"] = np.nan
        if "reviewDetailedRating" in item and "Service" in item["reviewDetailedRating"]:
            df.loc[i, "reviewDetailedRating/Service"] = item["reviewDetailedRating"]["Service"]
        else:
            df.loc[i, "reviewDetailedRating/Service"] = np.nan
        if "reviewDetailedRating" in item and "Atmosphere" in item["reviewDetailedRating"]:
            df.loc[i, "reviewDetailedRating/Atmosphere"] = item["reviewDetailedRating"]["Atmosphere"]
        else:
            df.loc[i, "reviewDetailedRating/Atmosphere"] = np.nan
        df.loc[i, "reviewerNumberOfReviews"] = item["reviewerNumberOfReviews"]
        df.loc[i, "text"] = item["text"]
        df.loc[i, "textTranslated"] = item["textTranslated"]
        df.loc[i, "stars"] = item["stars"]

    return df

if __name__ == "__main__":
    url = "https://www.google.com/maps/place/Schwabinger+Wassermann/@48.1628562,11.5728463,15z/data=!4m2!3m1!1s0x0:0xd26d3a34dedb378d?sa=X&ved=2ahUKEwjpu8ycyeuCAxUKO-wKHae5C5gQ_BJ6BAhNEAA"
    df = scrape_apify(url)
    print(df.head())
    print(df.stars.mean())
    print(df.columns)
