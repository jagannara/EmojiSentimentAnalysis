# Dataset Source: https://data.world/crowdflower/sentiment-analysis-in-text
print("Step 0: Load Preprocessed Data")
print("\n")

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

df = pd.read_csv("preprocessed_multiclass_data.csv")
to_predict = pd.read_csv("X_train.csv")

# consider rows without null "lemmatize_text" values only, where "lemmatize_text" refers to preprocessed text
df = df[df["lemmatize_text"].notna()]
to_predict = to_predict[to_predict["2"].notna()]

# remove numbers from text
df["lemmatize_text"] = df["lemmatize_text"].str.replace("\d+", "")

# based on heuristic evaluation, remove emotions that do not help provide accurate predictions and that have low counts
df = df[df.sentiment != "anger"]
df = df[df.sentiment != "boredom"]
df = df[df.sentiment != "enthusiasm"]
df = df[df.sentiment != "fun"]
df = df[df.sentiment != "hate"]
df = df[df.sentiment != "relief"]
df = df[df.sentiment != "surprise"]
df = df[df.sentiment != "empty"]
df = df[df.sentiment != "love"]

# TODO: Optional - merge classes, did not help improve accuracy of model
# df["sentiment"] = df[["happy", "sad", "worry", "neutral"]]
# # merging similar sentiment classes with low count together
# df["sentiment"].replace(
#     {
#         "anger": "hate",
#         "boredom": "neutral",
#         "empty": "neutral",
#         "fun": "happiness",
#         "enthusiasm": "happiness",
#     },
#     inplace=True,
# )

# label encoding
label_encoder = preprocessing.LabelEncoder()
df["sentiment_encoded"] = label_encoder.fit_transform(
    df["sentiment"]
)  # sentiment converted to numbers in the range of 0 to 12


print("Step 1: Inspect preprocessed text data")
print(df["lemmatize_text"][:5])
print("\n")


print("Step 2: View unique sentiment/mood classes")
print(df.sentiment.unique())
print("\n")


print("Step 3: Visualize dataset")
# Reference: https://www.kaggle.com/shainy/twitter-emotion-analysis

import matplotlib.pyplot as plt

sentiment_count = df.groupby("sentiment").count()
plt.bar(sentiment_count.index.values, sentiment_count["lemmatize_text"])
plt.xlabel("Tweet Sentiments")
plt.ylabel("Number of Tweets")
plt.show()
print("\n")

print("Step 4: Baseline Model - Logistic Regression Classifier")
# Reference: https://www.kaggle.com/vanshjatana/text-classification-from-scratch

# from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import (
    feature_extraction,
    linear_model,
    model_selection,
    preprocessing,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
)

x_train, x_test, y_train, y_test = train_test_split(
    df["lemmatize_text"],
    df["sentiment_encoded"],
    test_size=0.2,
    random_state=42,
)

pipe = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("model", LogisticRegression(max_iter=10000)),
    ]
)

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print(
    "Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100, 2))
)

print("Confusion Matrix:")
# Reference: https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
confusion_matrix = confusion_matrix(y_test, prediction)
plt.figure(figsize=(9, 9))
sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt=".3f",
    linewidths=0.5,
    square=True,
    cmap="Blues_r",
)
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix for Linear Regression", size=15)
plt.show()
print("\n")

print("Classification Report:")
print(classification_report(y_test, prediction))
print("\n")


# print("Step 5: Support Vector Classifier")
# Reference: https://www.kaggle.com/vanshjatana/text-classification-from-scratch

from sklearn.multiclass import OneVsRestClassifier

x_train, x_test, y_train, y_test = train_test_split(
    df["lemmatize_text"],
    df["sentiment_encoded"],
    test_size=0.2,
    random_state=42,
)

# use class_weight="balanced" to remove the imbalance in the data
model = Pipeline(
    [
        ("vectorizer", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", OneVsRestClassifier(LinearSVC(class_weight="balanced"))),
    ]
)

# using Grid Search to select the optimal input parameters
# Reference: https://towardsdatascience.com/a-production-ready-multi-class-text-classifier-96490408757
from sklearn.model_selection import GridSearchCV

parameters = {
    "vectorizer__ngram_range": [(1, 1), (1, 2), (2, 2)],
    "tfidf__use_idf": (True, False),
}
gs_clf_svm = GridSearchCV(model, parameters, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(x_train, y_train)
# print(gs_clf_svm.best_score_)
# print(gs_clf_svm.best_params_)

# updating the model using the results from Grid Searxh
model = Pipeline(
    [
        ("vectorizer", CountVectorizer(ngram_range=(1, 2))),
        ("tfidf", TfidfTransformer(use_idf=False)),
        ("clf", OneVsRestClassifier(LinearSVC(class_weight="balanced"))),
    ]
)

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print(
    "accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100, 2))
)
print("\n")


print("Step 6: Random Forest Classifier")
# Reference:
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("model", RandomForestClassifier()),
    ]
)

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print(
    "accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100, 2))
)
print("\n")


print("Step 7: Predicting New Data")

x_train, y_train = df["lemmatize_text"], df["sentiment_encoded"]
x_test = to_predict["2"]

pipe = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("model", LogisticRegression(max_iter=10000)),
    ]
)

model = pipe.fit(x_train, y_train)
to_predict["prediction"] = model.predict(x_test)

# write to CSV file
to_predict["labels"] = label_encoder.inverse_transform(
    to_predict["prediction"]
)
to_predict.rename(columns={"prediction": "X_train"}, inplace=True)
to_predict.to_csv("output.csv")
