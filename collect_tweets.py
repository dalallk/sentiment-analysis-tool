import tweepy
from dotenv import load_dotenv
import os
import json

load_dotenv()

BEARER_TOKEN = os.getenv('BEARER_TOKEN')

client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Fetch recent tweets
tweets = client.search_recent_tweets(query="#AI", tweet_fields=["author_id", "created_at"], max_results=50)

# Check if the 'data' attribute exists
if tweets.data:
    # Save tweets to a JSON file
    with open('tweets.json', 'w') as f:
        json.dump([tweet.data for tweet in tweets.data], f)  # Adjusted to handle the tweet data correctly

    print("Tweets saved successfully.")
else:
    print("No tweets found or error in fetching tweets.")
