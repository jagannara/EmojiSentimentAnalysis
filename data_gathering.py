# Step 1: List of CEO Twitter Handles of Interest
import pandas as pd

df = pd.read_csv("CEO_tweets.csv")
# Reference: https://realpython.com/python-csv/#parsing-csv-files-with-pythons-built-in-csv-library

twitterHandles = []
for index, row in df.iterrows():
    # remove whitespace in twitter handle data
    company = row["Company"].strip()
    handle = row["Twitter Handle"].strip()
    twitterHandles.append([handle, company])
print(twitterHandles)

# Reference: https://www.journaldev.com/23763/python-remove-spaces-from-string


# Step 2: Scraping Twitter for Tweets of Identified CEOs (2015 to 2020)
import snscrape.modules.twitter as sntwitter
import csv

# setting scope of extraction
# twitterHandles = twitterHandles[0:10]

# set minimum threshold for number of tweets
# minTweets = 300

csvFile = open("gathered_data.csv", "a", newline="", encoding="utf8")

# scrape id, date and text from every tweet
csvWriter = csv.writer(csvFile)
csvWriter.writerow(
    ["company", "CEO", "id", "date", "text",]
)

# filter tweets that are links, replies or RT - not organic tweets
for handle in twitterHandles:
    for i, tweet in enumerate(
        sntwitter.TwitterSearchScraper(
            "from:%s + since:2015-11-01 until:2020-11-01 -filter:links -filter:replies"
            % handle[0]
        ).get_items()
    ):
        # if i < minTweets:
        #     break
        csvWriter.writerow(
            [handle[1], handle[0], tweet.id, tweet.date, tweet.content]
        )
csvFile.close()

# Reference: https://github.com/JustAnotherArchivist/snscrape
