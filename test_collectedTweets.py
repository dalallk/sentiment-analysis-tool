import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # Correct import for joblib
import nltk

# Load necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define preprocessing function (ensure it's the same as when training your model)
def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    # Example preprocessing steps:
    text = text.lower()  # Lowercase
    tokens = word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords and non-alphabetic words
    return ' '.join(tokens)

# Load the processed tweets (make sure the CSV file has a 'text' column with the tweets)
processed_data = pd.read_csv('processed_tweets.csv')

# Preprocess the tweets
processed_data['processed_text'] = processed_data['text'].apply(preprocess_text)

# Load the trained TF-IDF vectorizer and model
tfidf_vectorizer = joblib.load('vectorizer.pkl')  # Replace with your actual vectorizer file path
model = joblib.load('sentiment_model.pkl')  # Replace with your actual model file path

# Transform the processed tweets using the same TF-IDF vectorizer
X_test = tfidf_vectorizer.transform(processed_data['processed_text'])

# Make predictions
predictions = model.predict(X_test)

# Add predictions to the dataframe
processed_data['predictions'] = predictions

# Save the results to a new CSV file
processed_data.to_csv('predicted_results.csv', index=False)

# Print the results
print(processed_data[['text', 'predictions']].head())
