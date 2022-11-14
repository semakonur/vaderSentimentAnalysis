#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tweepy
from textblob import TextBlob
import re
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
df = pd.read_csv('doguSıgınmacı.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)


# In[ ]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt
def clean_tweets(tweets):
    #rt sil (RT @xxx:)
    tweets = np.vectorize(remove_pattern)(tweets, "RT @[\w]*:") 
    
     
    #bahsedilenleri sil (@xxx)
    tweets = np.vectorize(remove_pattern)(tweets, "@[\w]*")
    
    #url link sil (httpxxx)
    tweets = np.vectorize(remove_pattern)(tweets, "https?://[A-Za-z0-9./]*")
    
    #özel karakterleri, sayıları, noktalama işaretlerini kaldırın 
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z]", " ")
   

    
    return tweets


# In[ ]:


df["tweets"] = df['tweets'].str.replace('[^\w\s]','')
df['tweets'] = df['tweets'].str.replace('\d+', '')


# In[ ]:


df['tweets'] = clean_tweets(df['tweets'])


# In[ ]:


scores = []
# Declare variables for scores
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
for i in range(df['tweets'].shape[0]):
#print(analyser.polarity_scores(sentiments_pd['text'][i]))
    compound = analyzer.polarity_scores(df['tweets'][i])["compound"]
    pos = analyzer.polarity_scores(df['tweets'][i])["pos"]
    neu = analyzer.polarity_scores(df['tweets'][i])["neu"]
    neg = analyzer.polarity_scores(df['tweets'][i])["neg"]
    
    
    scores.append({"Compound": compound,
                       "Positive": pos,
                       "Negative": neg,
                       "Neutral": neu
                   
                  })


# In[ ]:


sentiments_score = pd.DataFrame.from_dict(scores)
df = df.join(sentiments_score)


# In[ ]:


def getPolarity(twt):
    return TextBlob(twt).sentiment.polarity

df["Polarity"]=df["tweets"].apply(getPolarity)
def format_output(output_dict):
    if(output_dict['compound']> 0):
        return "Positive"

    elif(output_dict['compound']< 0):
        return "Negative"
    else:
        return "Neutral"  
def predict_sentiment(text):
  
  output_dict =  analyzer.polarity_scores(text)
  return format_output(output_dict)

df["vader_prediction"] = df["tweets"].apply(predict_sentiment)

df


# In[ ]:


def format_output(output_dict):

    if(output_dict['compound']> 0):
        return "Positive"

    elif(output_dict['compound']< 0):
        return "Negative"
    else:
        return "Neutral"
    
  
def predict_sentiment(text):
  
  output_dict =  analyzer.polarity_scores(text)
  return format_output(output_dict)

# Run the predictions
df["Polarity"] = df["tweets"].apply(predict_sentiment)

# Show 5 random rows of the data
df


# In[ ]:


df["vader_prediction"].value_counts().plot(kind='pie')
plt.title("Sentiment Analysis Pie Plot")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.show()


# In[ ]:



df["vader_prediction"].value_counts().plot(kind='pie',colors=["g", "r", "c"],
    autopct="%.2f",
    fontsize=20,
    figsize=(6, 6),)
plt.title("Sentiment Analysis Pie Plot")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(df['Polarity'], df['vader_prediction'])

print("Accuracy: {}\n".format(accuracy))

# Show the classification report
print(classification_report(df['Polarity'], df['vader_prediction']))


# In[ ]:


def word_cloud(wd_list):
    stopwords = set(STOPWORDS)
    all_words = ' '.join([text for text in wd_list])
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        width=1600,
        height=800,
        random_state=1,
        colormap='jet',
        max_words=80,
        max_font_size=200).generate(all_words)
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation="bilinear");
word_cloud(df['tweets'])


# In[ ]:




