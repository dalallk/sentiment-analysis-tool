import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources if you haven't already
nltk.download('punkt')
nltk.download('stopwords')

# Load the JSON file containing tweets (make sure the path is correct)
with open('tweets.json', 'r') as f:
    tweets_data = json.load(f)

# Convert JSON to a pandas DataFrame
tweets = pd.DataFrame(tweets_data)

# Preview the data to see what the structure looks like
print(tweets.head())

# Preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(filtered_tokens)

# Apply preprocessing to the 'text' column (adjust the column name if necessary)
tweets['processed_text'] = tweets['text'].apply(preprocess_text)

# Preview the cleaned data
print(tweets[['text', 'processed_text']].head())

# Save the processed tweets to a CSV file
tweets.to_csv('processed_tweets.csv', index=False)

print("Preprocessing complete. Processed data saved to 'processed_tweets.csv'.")
