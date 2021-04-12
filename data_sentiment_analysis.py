# Part A - Text Classification with TextBlob

print("Part A - Text Classification with TextBlob")
print("\n")

import numpy as np
import pandas as pd
from textblob import TextBlob

# Step 0: Load Gathered Data For Cleaning
df = pd.read_csv("preprocessed_data.csv", lineterminator="\n")
print("Step 0: Before Classification")
print(df.head())
print("\n")

# Step 1: Get Subjectivity and Polarity Scores
# Subjectivity of tweet (how opinionated it is, 0 = Fact -> 1 = Opinion)
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# Polarity of tweet (how positive or negative it is, -1 = Highest Negative Score -> +1 = Highest Positive Score)
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


df["lemmatize_text"] = df.lemmatize_text.astype(str)
df["subjectivity_score"] = df["lemmatize_text"].apply(
    lambda x: getSubjectivity(x)
)
df["polarity_score"] = df["lemmatize_text"].apply(lambda x: getPolarity(x))
print("Step 1: Get Subjectivity and Polarity Scores")
print(df.head())
print("\n")

# Step 2: Classify Polarity Score into Positive, Negative or Neutral
def getAnalysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"


df["class"] = df["polarity_score"].apply(lambda x: getAnalysis(x))
print("Step 2: Classify Polarity Score into Positive, Negative or Neutral")
print(df.head())
print("\n")

# exporting to CSV file
df.to_csv("textblob_classification_data.csv", index=False)


# TODO: Part B - Multi-Class Classification with SentiWordNet

# TODO: Part C - Multi-Class Classification with Own Classes (Angry, Calm, Happy, Sad)

