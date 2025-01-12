import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary resources (only needed once)
nltk.download('stopwords')
nltk.download('punkt')

# Load the Sentiment140 dataset
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1')  # Replace with the correct filename and encoding if necessary

# Rename columns if the dataset uses unnamed ones (common in Sentiment140)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']  # Adjust column names as needed

# Function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (e.g., #hashtag)
    text = re.sub(r'#\w+', '', text)
    # Remove non-alphabetic characters (optional)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase to maintain consistency
    text = text.lower()
    return text

# Apply cleaning function
df['cleaned_text'] = df['text'].apply(clean_text)

# Tokenize the text
df['tokenized_text'] = df['cleaned_text'].apply(word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['filtered_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Apply stemming (or lemmatization if you prefer)
stemmer = PorterStemmer()
df['stemmed_text'] = df['filtered_text'].apply(lambda x: [stemmer.stem(word) for word in x])

# Join the words back into a single string
df['processed_text'] = df['stemmed_text'].apply(lambda x: ' '.join(x))

# Save the processed data to a new CSV file
df.to_csv('processed_sentiment140.csv', index=False)

# Display the first few rows of the original and processed text for verification
print(df[['text', 'processed_text']].head())
