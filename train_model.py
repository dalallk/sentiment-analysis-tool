import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the preprocessed dataset
df = pd.read_csv('processed_sentiment140.csv')  # Ensure the correct path to your file

# Check for missing values and drop them if necessary
df = df.dropna(subset=['processed_text', 'target'])

# Quick check to ensure correct columns
print(df[['processed_text', 'target']].head())

# Set the feature and target variables
X = df['processed_text']  # Text data
y = df['target']          # Sentiment labels (0 = negative, 2 = neutral, 4 = positive)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical format using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and vectorizer for future use (optional)
import joblib
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')