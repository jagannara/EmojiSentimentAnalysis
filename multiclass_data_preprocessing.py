# Part A - Text Cleaning
# Reference: https://www.kaggle.com/longtng/nlp-preprocessing-feature-extraction-methods-a-z

print("Part A - Text Cleaning")
print("\n")

import numpy as np
import pandas as pd

# Step 0: Load Gathered Data For Cleaning
df = pd.read_csv("text_emotion.csv")
print("Step 0: Before Cleaning")
print(df.head())
print("\n")


# Step 1: Convert Emoji into Words
# TODO: Next experiment: Keep emojis as is, and assign sentiment points to emojis instead of converting to words
# Reference: https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing
# pip3 install emot

import emot
import re

from emot.emo_unicode import UNICODE_EMO, EMOTICONS


def convert_emojis(text):
    for emot in UNICODE_EMO:
        if emot in text:
            text = re.sub(
                r"(" + emot + ")",
                "_".join(
                    UNICODE_EMO[emot].replace(",", "").replace(":", "").split()
                ),
                text,
            )
    return text


df["emoji_to_words"] = df["content"].apply(lambda x: convert_emojis(x))
print("Step 1: Convert Emoji into Words")
print(df.head())
print("\n")


# Step 2: Upper Case to Lower Case Conversion
df["clean_text"] = df["emoji_to_words"].apply(lambda x: x.lower())
print("Step 2: Upper Case to Lower Case Conversion")
print("Before:")
print(df["content"][1:5])
print("After:")
print(df["emoji_to_words"][1:5])
print("\n")


# Step 3: Expand Contractions
import contractions

df["clean_text"] = df["clean_text"].apply(lambda x: contractions.fix(x))
print("Step 3: Expand Contractions")
print("Before:")
print(df["content"][182])
print("After:")
print(df["clean_text"][182])
print("\n")

# Step 4: Noise Removal (keep emojis, symbols and graphic characters)

# remove html tags
def remove_html(text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)


# remove urls
def remove_url(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)


# remove non-ascii characters
def remove_non_ascii(text):
    return re.sub(r"[^\x00-\x7f]", r"", text)


# remove punctuation
def remove_punctuation(text):
    return re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", text)


def manual_cleaning(text):
    # TODO: fix slangs, abbreviations based on observations
    return None


df["clean_text"] = df["clean_text"].apply(lambda x: remove_html(x))
df["clean_text"] = df["clean_text"].apply(lambda x: remove_url(x))
df["clean_text"] = df["clean_text"].apply(lambda x: remove_non_ascii(x))
df["clean_text"] = df["clean_text"].apply(lambda x: remove_punctuation(x))

print("Step 4: Noise Removal")
print("Before:")
print(df["content"][213])
print("After:")
print(df["clean_text"][213])
print("\n")

# TODO: Step 4: Spelling Correction (Manual)
# sample automatic correction, with risk of errors
# from textblob import TextBlob
# df["clean_text"] = df["content"].apply(lambda x: TextBlob(x).correct())


# Part B - Text Preprocessing

print("Part B - Text Preprocessing")
print("\n")

# Step 1: Tokenization
from nltk.tokenize import word_tokenize

df["tokens"] = df["clean_text"].apply(word_tokenize)
print("Step 1: Tokenization")
print(df.head())
print("\n")

# Step 2: Remove Stop Words (e.g. "in", "then", "the")
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

stopWords = set(stopwords.words("english"))
df["remove_stopwords"] = df["tokens"].apply(
    lambda x: [word for word in x if word not in stopWords]
)
print("Step 2: Remove Stop Words")
print(df.tail())
print("\n")

# Step 3: Stemming
# TODO: reduces accuracy, consider removing
# using SnowballStemmer - turned "happy" to "happi"
# if used, needs POS and lemmatization

from nltk.stem import SnowballStemmer


def stemmer(text):
    snowballStemmer = nltk.SnowballStemmer("english")
    stems = [snowballStemmer.stem(i) for i in text]
    return stems


df["snowball_stemmer"] = df["remove_stopwords"].apply(lambda x: stemmer(x))
print("Step 3: Stemming with Snowball Stemmer")
print(df.tail())
print("\n")

# Step 4: POS Tagging (Part of Speech)
nltk.download("brown")
from nltk.corpus import wordnet
from nltk.corpus import brown

wordnet_map = {
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "J": wordnet.ADJ,
    "R": wordnet.ADV,
}

train_sents = brown.tagged_sents(categories="news")
t0 = nltk.DefaultTagger("NN")
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)


def pos_tag(text, pos_tag_type="pos_tag"):
    pos_tagged_text = t2.tag(text)
    pos_tagged_text = [
        (word, wordnet_map.get(pos_tag[0]))
        if pos_tag[0] in wordnet_map.keys()
        else (word, wordnet.NOUN)
        for (word, pos_tag) in pos_tagged_text
    ]
    return pos_tagged_text


df["pos_tagged"] = df["remove_stopwords"].apply(lambda x: pos_tag(x))
print("Step 4: POS Tagging")
print(df.tail())
print("\n")

# Step 5: Lemmatization
from nltk.stem import WordNetLemmatizer


def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(word, tag) for word, tag in text]
    return lemma


df["lemmatize"] = df["pos_tagged"].apply(lambda x: lemmatize(x))
# double check to remove stop words
df["lemmatize"] = df["lemmatize"].apply(
    lambda x: [word for word in x if word not in stopWords]
)
# join back to text
df["lemmatize_text"] = [" ".join(map(str, l)) for l in df["lemmatize"]]
print("Step 5: Lemmatize")
print(df.tail())
print("\n")

# TODO: Step 6: Language Detection + Removal of Foreign Language (if necessary)
# pip3 install pyicu
# pip3 install pycld2
# pip3 install polyglot

# Part C
# Reference: https://www.geeksforgeeks.org/saving-a-pandas-dataframe-as-a-csv/

# exporting to CSV file
df.to_csv("preprocessed_multiclass_data.csv", index=False)

