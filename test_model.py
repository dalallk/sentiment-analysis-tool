import joblib

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define a sample input for testing
sample_texts = [
    "I love this product! It's amazing.",
]

# Preprocess and vectorize the sample input
sample_vectorized = vectorizer.transform(sample_texts)

# Predict sentiments
predictions = model.predict(sample_vectorized)

# Map target values to sentiment labels (optional)
sentiment_map = {0: "Negative", 2: "Neutral", 4: "Positive"}
predicted_labels = [sentiment_map[pred] for pred in predictions]

# Print results
for text, sentiment in zip(sample_texts, predicted_labels):
    print(f"Text: {text} -> Sentiment: {sentiment}")
