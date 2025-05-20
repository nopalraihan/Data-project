# Data-project
# Sentiment Analysis of the Film Deadpool and Wolverine Using Twitter API

#### Project Objective
- **Identify the sentiment of the film Deadpool and Wolverine in Indonesia.**
  - Platform: Twitter/X.

#### Data Collection
- **Using Twitter/X API**
  - I created a Twitter Developer account and generated API keys to automatically fetch hashtag data using Python.
  - **Code to collect Twitter data:**
    ```python
    import tweepy

    # Insert your API keys and tokens
    consumer_key = 'your_consumer_key'
    consumer_secret = 'your_consumer_secret'
    access_token = 'your_access_token'
    access_token_secret = 'your_access_token_secret'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # Collect tweets based on hashtag
    hashtag = "#DeadpoolAndWolverine"
    tweets = tweepy.Cursor(api.search_tweets, q=hashtag, lang="id", tweet_mode='extended').items(1000)

    # Store tweets in a list
    tweet_data = []
    for tweet in tweets:
        tweet_data.append(tweet.full_text)

    print("Number of tweets collected:", len(tweet_data))
    ```

#### Data Pre-processing
- **Clean text data**
  - Remove punctuation, links, hashtags, and usernames.
  - Convert text to lowercase.
  - Remove stopwords (common words that do not provide significant meaning like "and", "in", etc.).
  - **Example code for data pre-processing:**
    ```python
    import re
    from nltk.corpus import stopwords

    # Function to clean tweets
    def clean_tweet(tweet):
        tweet = re.sub(r'http\S+', '', tweet)  # Remove links
        tweet = re.sub(r'@\w+', '', tweet)     # Remove usernames
        tweet = re.sub(r'#\w+', '', tweet)     # Remove hashtags
        tweet = re.sub(r'[^\w\s]', '', tweet)  # Remove punctuation
        tweet = tweet.lower()                  # Convert text to lowercase
        tweet = ' '.join(word for word in tweet.split() if word not in stopwords.words('indonesian'))
        return tweet

    # Apply the function to all tweets
    clean_tweets = [clean_tweet(tweet) for tweet in tweet_data]
    print(clean_tweets[:5])
    ```

#### Sentiment Analysis
- **Using Pre-trained Model or Sentiment Library**
  - Use libraries like TextBlob or VADER for simple sentiment analysis.
  - **Code with TextBlob:**
    ```python
    from textblob import TextBlob

    # Function to get sentiment
    def get_sentiment(tweet):
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    # Apply the function to all tweets
    sentiments = [get_sentiment(tweet) for tweet in clean_tweets]
    print(sentiments[:5])
    ```

#### Save Data for Visualization
- **Save Processed Data and Sentiment Analysis Results to CSV**
  - Save the processed data and sentiment analysis results to a CSV file for use in Tableau and Power BI.
  - **Code to save data:**
    ```python
    import pandas as pd

    # Create DataFrame
    df = pd.DataFrame({
        'tweet': tweet_data,
        'clean_tweet': clean_tweets,
        'sentiment': sentiments
    })

    # Save to CSV file
    df.to_csv('sentiment_analysis.csv', index=False)
    ```

### Data Visualization with Pandas and Matplotlib

The code below is used to display the sentiment distribution from the collected tweet data using the Twitter API.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('sentiment_analysis.csv')

# Display the first few rows to ensure data is loaded correctly
print(data.head())

# Create a bar chart for sentiment distribution
sentiment_counts = data['sentiment'].value_counts()
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'blue', 'red'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=0)
plt.grid(axis='y')

# Show the plot
plt.show()
```

![Sentiment Distribution](https://github.com/byf1sh/Data-Analysis-Project/blob/main/Sentiment%20Analysis%20Of%20the%20Film%20Deadpool%20And%20Wolverine/Assets/Positive-Negative-Sentiment.png?raw=true)

The code below is used to find the average occurrence of specific words in user tweets for further analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
file_path = 'sentiment_analysis.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Remove rows with NaN values in 'clean_tweet' column
data = data.dropna(subset=['clean_tweet'])

# Words to count frequency
words_to_count = ["best", "exciting", "good", "cool", "epic", "amazing"]

# Count the frequency of words
word_counts = {word: 0 for word in words_to_count}
for tweet in data['clean_tweet']:
    if isinstance(tweet, str):
        for word in words_to_count:
            word_counts[word] += tweet.split().count(word)

# Create a DataFrame from the word counts
word_count_df = pd.DataFrame(list(word_counts.items()), columns=['word', 'count'])

# Display the DataFrame to ensure correct results
print(word_count_df)

# Create a bar chart for word frequency
plt.figure(figsize=(10, 6))
plt.bar(word_count_df['word'], word_count_df['count'], color='skyblue')
plt.title('Word Frequency in Tweets')
plt.xlabel('Word')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Show the plot
plt.show()
```

![Word Frequency](https://github.com/byf1sh/Data-Analysis-Project/blob/main/Sentiment%20Analysis%20Of%20the%20Film%20Deadpool%20And%20Wolverine/Assets/word%20count.png?raw=true)

Thankyou.
