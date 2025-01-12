import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load preprocessed data
df = pd.read_csv('processed_sentiment140.csv')

# Check for missing values in 'processed_text'
missing_values = df['processed_text'].isnull().sum()
print(f"Number of missing values in 'processed_text': {missing_values}")

# Drop rows with missing 'processed_text'
df = df.dropna(subset=['processed_text'])

# Verify the changes
print(f"Number of rows after dropping missing values: {len(df)}")

# Define features and target
X = df['processed_text']
y = df['target']

# Vectorize text data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved successfully!")
