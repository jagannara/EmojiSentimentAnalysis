# EmojiSentimentAnalysis

This repo applies sentiment analysis and machine learning principles to find a correlation
between “CEO tweet sentiment” and “stock price movement.” Using public Twitter data of 49
CEOs over 5 years, I classified their tweets into mood states that were then used to predict the
stock price movement of their respective companies. This work expands on traditional sentiment
analysis by emphasizing the role of emojis, where present, in the mood representation of each
tweet. I found that particular mood states such as “happiness” have a strong correlation with
stock market movement and can be used to predict stock prices.

The following steps/methods were followed: 
1. Gathering Twitter Data (see data_gathering.py)
2. Text Cleaning and Pre-Processing (see data_preprocessing) : The tweet text data was cleaned by turning uppercase letters to lowercase, expanding contractions, removing noise such as URLs, punctuation, and correcting spelling errors. The text was then preprocessed by tokenization, removal of stop words, Snowball stemming, Part-Of-Speech tagging, and lemmatization.
3. Basic Tweet Sentiment Analysis using Emoji Analysis (see data_sentiment_analysis.py): A. This expands upon existing mood assessment tools by incorporating a more in-depth “emoji” analysis, where the impact of using different types of emoji characters at different frequencies on stock market sentiment is evaluated. Using an existing rule-based sentiment analysis library, TextBlob, the polarity and subjectivity of each tweet was obtained. Based on the resulting polarity scores that ranged from -1 to 1, we classified each tweet as positive (polarity > 0.5) , negative (polarity < -0.5) or neutral.
4. Advanced Tweet Sentiment Analysis using Mood Classification (see multiclass_data_preprocessing.py and multiclass_data_classification.py): Tweets were classified into 13 different mood classes e.g. angry, happy, worried, sad, enthusiastic. I used a One-vs-Rest implementation of SVM to model a publicly available, labelled Twitter dataset, consisting of 48,000 tweets and their human-annotated moods. Additional weighting was assigned to emojis using TF-IDF. I then used this model to predict mood classes for the tweet data. Existing multiclass classification tools like SentiWordnet did not allow me to effectively weigh the sentiment of emojis, which is why I had to build my own model.
5. Correlating Tweet Sentiment Analysis with Stock Price Movement: For every tweet from Phase 1, I classified the corresponding one-day stock price movement as an “Increase” or “Decrease”. In case the stock market was not open on the day of the tweet (i.e. weekend), I used the price from the nearest day when it was open. For the baseline model, I trained a logistic regression model due its binomial classification nature and low computing power requirement. In later experiments I investigated the use of other classifications
algorithms such as Ridge regression and SVM, and conducted parameter tuning using heuristics and grid search.

