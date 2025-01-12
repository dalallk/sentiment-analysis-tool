import tweepy
from dotenv import load_dotenv
import os

load_dotenv()

BEARER_TOKEN = os.getenv('BEARER_TOKEN')

# Authenticate using Bearer Token for read-only access
client = tweepy.Client(bearer_token=BEARER_TOKEN)

user = client.get_me()  # Verify the authentication
print(f"Authenticated as {user.data['username']}")