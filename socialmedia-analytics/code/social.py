import social_tests as test

### PHASE 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def parse_label(label):
    d = dict()
    n_start = label.find(" ")
    n_end = label.find("(")
    p_end = label.find("from")
    name = label[n_start:n_end].strip()
    position = label[n_end+1:p_end].strip()
    state = label[p_end+len(" from"):-1]
    d["name"] = name
    d["position"] = position
    d["state"] = state
    return d 

def get_region_from_state(state_df, state):
    for i in range(len(state_df)):
        if state_df.iloc[i]["state"] == state:
            return (str(state_df.iloc[i]["region"]))

end_chars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]
def find_hashtags(message):
    lst = []
    for i in range(len(message)):
        if message[i] == "#":
            result = message[i]
            j = i+1
            while j < len(message) and message[j] not in end_chars:
                result += message[j]
                j += 1
            lst.append(result)
            result = ""
    return lst

def find_sentiment(classifier, message):
    sentiment = classifier.polarity_scores(message)
    if sentiment["compound"] > 0.1:
        return (sentiment["compound"], "positive")
    elif sentiment["compound"] < -0.1:
        return (sentiment["compound"], "negative")
    else:
        return (sentiment["compound"], "neutral")
    
def add_columns(data, state_df):
    classifier = SentimentIntensityAnalyzer()
    names = []
    positions = []
    states = []
    regions = []
    for label in data["label"]:
        result = parse_label(label)
        names.append(result["name"])
        positions.append(result["position"])
        state = result["state"]
        states.append(state)
        region = get_region_from_state(state_df,state)
        regions.append(region)
    data["name"] = names
    data["position"] = positions
    data["state"] = states
    data["region"] = regions

    hashtags, scores, sentiments = [], [], []
    for text in data["text"]:
        hashtags.append(find_hashtags(text))
        (score, category) = find_sentiment(classifier, text)
        scores.append(score)
        sentiments.append(category)
    data["hashtags"] = hashtags
    data["score"] = scores
    data["sentiment"] = sentiments
    return

### PHASE 2 ###

def get_sentiment_quantiles(data, col_name, col_value):
    if col_name != "":
        data = data[data[col_name] == col_value]
    result = [data["score"].min()]
    result.extend(list(round(data["score"].quantile([0.25, 0.5, 0.75]),5)))
    result.append(data["score"].max())
    print(result)
    return result

def get_hashtag_subset(data, col_name, col_value):
    data = data[data[col_name] == col_value]
    all_hashtags = set()
    for hashtags in data["hashtags"]:
        for tag in hashtags:
            all_hashtags.add(tag)
    return all_hashtags

def get_hashtag_rates(data):
    d = {}
    for hashtags in data["hashtags"]:
        for tag in hashtags:
            if tag not in d:
                d[tag] = 0
            d[tag] += 1
    return d

def most_common_hashtags(hashtags, count):
    best_only = {}
    while len(best_only) < count:
        curr_best = None
        curr_count = 0
        for k in hashtags:
            if hashtags[k] > curr_count and k not in best_only:
                curr_best = k
                curr_count = hashtags[k]
        best_only[curr_best] = curr_count
    return best_only

def get_hashtag_sentiment(data, hashtag):
    total = 0
    count = 0
    for index,row in data.iterrows():
        hashtags = row["hashtags"]
        sent = row["sentiment"]
        if hashtag in hashtags:
            count += 1
            if sent == "positive":
                total += 1
            elif sent == "negative":
                total -= 1
    return total / count

### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    test.test_all()
    test.run()