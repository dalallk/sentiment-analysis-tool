# Sentiment Analysis Tool

The **Sentiment Analysis Tool** is a project designed to analyze text sentiments using machine learning models. The tool processes data, trains a model, and predicts sentiments for given text inputs such as tweets or product reviews. If you'd like to try it out and contribute, follow the setup instructions below.

## Features

- **Data Collection**: Collect tweets using the Twitter API with Tweepy.
- **Data Preprocessing**: Clean and preprocess text data to prepare it for training.
- **Model Training**: Train a sentiment analysis model using a Naive Bayes classifier.
- **Sentiment Prediction**: Predict sentiments for new text inputs or pre-collected tweets.
- **Export Results**: Save predictions to CSV for further analysis.

## File Overview

### Authentication
- **`twitter_auth.py`**  
  Authenticates the Tweepy client using a bearer token from environment variables.

### Data Collection
- **`collect_tweets.py`**  
  Fetches recent tweets using a specific query and saves them to a JSON file.

### Data Preprocessing
- **`preprocess_data.py`**  
  Prepares the Sentiment140 dataset by cleaning, tokenizing, and stemming text data. Outputs the processed dataset to a CSV file.

- **`preprocess_tweets.py`**  
  Cleans and preprocesses tweets collected via the Twitter API. Saves the processed tweets to a CSV file.

### Model Training
- **`train_model.py`**  
  Trains a Naive Bayes classifier on the preprocessed Sentiment140 dataset and evaluates its performance. Saves the trained model and vectorizer.

- **`save_model.py`**  
  Ensures the dataset is cleaned, vectorized, and used for training. Saves the model and vectorizer for reuse.

### Testing
- **`test_model.py`**  
  Loads the trained model and vectorizer to predict sentiments for sample inputs.

- **`test_collectedTweets.py`**  
  Preprocesses collected tweets, predicts their sentiments using the trained model, and saves the predictions to a CSV file.

### Utilities
- **`download_nltk.py`**  
  Downloads necessary NLTK resources (e.g., stopwords, tokenizers) for preprocessing.

## Setup Instructions

To try out the project, follow these steps to set it up locally:

1. **Clone the Repository**  
   ```bash
   git clone <repository-url>
   cd sentiment-analysis-tool
   ```

2. **Install Dependencies**  
   Install the required Python libraries by running:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**  
   Create a `.env` file and add your Twitter API Bearer Token:  
   ```
   BEARER_TOKEN=your_bearer_token
   ```

4. **Prepare Datasets**  
   - Download the [Sentiment140 dataset](http://help.sentiment140.com/for-students/) and save it as `training.1600000.processed.noemoticon.csv` in the project directory.
   - Run `preprocess_data.py` to preprocess the Sentiment140 dataset.

5. **Collect Tweets (Optional)**  
   - If you'd like to collect real-time tweets, run `collect_tweets.py` to fetch tweets matching a specific query.  
   - Run `preprocess_tweets.py` to clean and preprocess the collected tweets.

6. **Train the Model**  
   Run `train_model.py` to train the sentiment analysis model on the preprocessed Sentiment140 dataset.

7. **Test the Model**  
   Use either `test_model.py` to test the model with sample inputs or `test_collectedTweets.py` to test the model on collected tweets.

## Results

- **Model Performance**: After training, the model's accuracy and classification report will be displayed.
- **Predicted Results**: Processed tweets with sentiment predictions are saved to `predicted_results.csv` for further analysis.

## Tools and Libraries Used

- **Python**: Core programming language.
- **Tweepy**: To interact with the Twitter API.
- **Pandas**: For data manipulation.
- **NLTK**: For natural language processing.
- **Scikit-learn**: For model training and evaluation.
- **Joblib**: To save and load models and vectorizers.

