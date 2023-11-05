#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 10:57:58 2022

@author: danielkotas


This script downloads Tweets of Jim Cramer, a finance TV-personality famous for being consistently wrong
in his stock recommendations.

Subsequently, the script analyses which stock (if any) is being mentioned in a given tweet and
assigns a sentiment to determine going long/short the stock.

Lastly, a simple trading strategy is devised based on Jim's recommendations.

Used resources:

Scraping Twitter:
https://medium.com/analytics-vidhya/twitter-scrapping-using-python-55a466b2f597
https://betterprogramming.pub/how-to-scrape-tweets-with-snscrape-90124ed006af

Troubleshooting:
https://github.com/JustAnotherArchivist/snscrape/issues/846

"""
# %% Imports

import snscrape.modules.twitter as sntwitter
import numpy as np
import ffmpeg
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import stanza
# stanza.download('en') # download English model
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
import nltk
# nltk.download(["names","stopwords","state_union","twitter_samples","movie_reviews", "averaged_perceptron_tagger","vader_lexicon","punkt" ])

from pprint import pprint
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pickle
import _pickle as cPickle
import bz2
import yfinance as yf
import requests
import bs4 as bs
from datetime import datetime
import yaml

sys.path.insert(0, '/Users/danielkotas/Documents/Documents – Daniel’s MacBook Air/Important /Extra Learning/modules')
from portfolio_analytics_v2 import PortfolioAnalysis


# %% Support functions


def compressed_pickle(title, data):
# compress data to a pickle file
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)

def decompress_pickle(file):
# load any compressed pickle file
 data = bz2.BZ2File(file, 'rb')
 data_load = cPickle.load(data)
 return data_load


def tp_date(df, leg, tp):
# calculates when to terminate a position based on desired take-profit as %-age
    if not df.empty:
        # provide a df with returns starting at a desired date
        nav = ((1+df).cumprod() - 1)*leg
        # calculate cumulative performance and multiply depending whether we want to long/short the stock
        return (nav > tp).argmax() # return index value of first record where the cum. perf is greater than set take profit
    else:
        return 0

def weights_scaler(df, long_cap, short_cap):
# Scales weights to adhere to budget and leverage constraints

    df_scaled = df.copy()
    df_scaled[df > 0] = df.div(df[df > 0].sum(axis=1),axis=0) * long_cap
    df_scaled[df < 0] = (df.div(df[df < 0].sum(axis=1),axis=0)) * short_cap
    return df_scaled

def next_working_day(date):
# takes a date and returns the next working day if the date is on a Saturday or Sunday
    if date.weekday() in [5, 6]:
        return date + pd.Timedelta(7 - date.weekday(), unit = 'D')
    return date

def save_sp500_tickers():
# finds names and tickers of SP500 companies
# original code by Yao Lei Xu: https://towardsdatascience.com/stock-market-analytics-with-pca-d1c2318e3f0e

    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'html')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    names = []
    forbidden = [' Inc.', ', .Inc']

    # go through table on the Wiki page
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        name = row.findAll('td')[1].text

        # replace "." with "-" per Yahoo convention
        ticker = ticker.replace(".", "-")
        for char in forbidden:
            name = name.replace(char, "")
        # apend individual ticker & name to aggregate list
        tickers.append(ticker.replace('\n', ''))
        names.append(name.replace('\n', ''))

    # create final dataframe to be returned
    df = pd.DataFrame(
        {'ticker': tickers,
         'name': names})
    return df

# %%  Main functions

def sent_class(text, nltk=True, stanza = True, textblob = True, textblob_subjectivity = True, method = 'avg'):
    # initiate empty list for storing sentiment scores
    overall_sentiment = []

    # classify sentiment with NLTK and append if set to true
    if nltk:
        sia = SentimentIntensityAnalyzer()
        nltk_sent = sia.polarity_scores(text)
        overall_sentiment.append(nltk_sent['compound'])

    # classify sentiment with stanza and append if set to true
    if stanza:
        stanza_raw = ()
        doc = nlp_stanza(text)
        for i, sentence in enumerate(doc.sentences):
            stanza_raw += (sentence.sentiment,)
        stanza_sent = np.mean(stanza_raw) -1 # stanza classifies sentiment as 0,1,2. We want -1,0,1
        overall_sentiment.append(stanza_sent)

    # # classify sentiment with textblob and append if set to true
    if textblob:
        tb_sent = TextBlob(text).sentiment.polarity
        if textblob_subjectivity:
            tb_sent = tb_sent * TextBlob(text).sentiment.subjectivity # weighting subjectivity
        overall_sentiment.append(tb_sent)

    # if method average, calculate simple mean of available sentiment
    if method == 'avg':
        final_sent = np.mean(overall_sentiment)
    # if method "max", take the single biggest value with the correct sign as the final sentiment
    elif method == 'max':
        if abs(np.min(overall_sentiment)) > abs (np.max(overall_sentiment)):
            final_sent = np.min(overall_sentiment)
        else:
            final_sent = np.max(overall_sentiment)

    return final_sent


def asg_ticker(t, tickers_names):
    tickers_names = tickers_names.fillna('#N/A - Missing Ticker')
    # empty list for returning found tickers
    found_tickers = []
    # clean tweet from https link first
    for char in t.split():
        if len(char) > 5 and char[:5] == 'https':
            t = t.replace(char, " ")
    # clean a tweet from commas, exclamation marks, numbers etc.
    disallowed_characters = "._!&,-?'0123456789();/`’"
    for char in disallowed_characters:
        t = t.replace(char, " ")
    # splitting words in a tweet
    t_split = t.split()
    # Option 1: finding stocks which begin with "$" with list comprehension
    tick = [x[1:] for x in t_split if x[0] == "$" and len (x) != 1]
    found_tickers += tick
    # Option 2: finding stocks by ticker but without the "$" sign
    # disallowing ticker 'A' because it is found often as an article "a"
    tick_2 = [x for x in t_split if x in list(tickers_names['ticker']) and x != 'A']
    found_tickers += tick_2
    # Option 3: finding stocks by name / alternative name
    for index, row in tickers_names.iterrows():
        if (row['name'] in t) or (row['alt_name'] in t) or (row['alt_name_2'] in t):
            found_tickers.append(row['ticker'])
    if found_tickers:
        return list(set(found_tickers))






# %% Parameters - loading .yaml file
with open("params.yaml", "r") as f:
    p = yaml.safe_load(f)
# %% Twitter scrape

if p['load_twitter']:
    # Creating list to append tweet data to
    tweets_list = []
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper('from:jimcramer').get_items()):
        print(i)
        if tweet.date.date() < datetime.strptime(p['start_date'],'%Y-%m-%d').date():
            break
        tweets_list.append([tweet.date,
                            tweet.id,
                            tweet.content,
                            tweet.user.username])
    # save newly scraped tweets to a pickle file
    compressed_pickle('tweets_cramer', tweets_list)
else:
    # if not loading new tweets, use existing database
    tweets_list = decompress_pickle('tweets_cramer.pbz2')

# Creating a dataframe from the tweets list above
tweets_df = pd.DataFrame(tweets_list,
                          columns=['Datetime',
                                   'Tweet Id',
                                   'Text',
                                   'Username'])


# %% Stock data

tickers_sp500 = save_sp500_tickers()
df_alt_names = pd.read_excel('alt_names.xlsx')
tickers_new = tickers_sp500.merge(df_alt_names,
                                  on='ticker',
                                  how='outer',
                                  sort=True).set_index('ticker', drop=False)

# define end date as the latest date in tweets
end_date = pd.to_datetime(tweets_df.iloc[0,:]['Datetime'], format='%Y-%m-%d').date()

if p['load_yahoo']:
    prices = yf.download(list(tickers_new.ticker), start = p['start_date'], end=end_date)['Adj Close']
    compressed_pickle('prices', prices)
else:
    prices = decompress_pickle('prices.pbz2')

prices.index = pd.to_datetime(prices.index, format = '%Y-%m-%d')
df_returns = prices.pct_change()


# %% Assigning stocks

tweets_asg = tweets_df.copy()
# note: runs around 4m on 30k tweets
tweets_asg['assign'] = tweets_asg['Text'].apply(lambda x: asg_ticker(x, tickers_new))


# %% Sentiment

tweets_sent = tweets_asg[tweets_asg['assign'].notnull()]
tweets_sent['sent'] = tweets_asg['Text'].apply(lambda x: sent_class(x, **p['sent_params']))

# replace prominent FB ticker with new "META" ticker
tweets_sent['assign'] = tweets_sent['assign'].apply(lambda lst: [x if x != 'FB' else 'META' for x in lst])


# %% Strategy

with open("params.yaml", "r") as f:
    p = yaml.safe_load(f)


# create a dataframe for creating cleaned dates
tweets_clean = tweets_sent.copy()

# trasnform dates of the tweets to pure datetime
tweets_clean['Datetime'] = tweets_clean['Datetime'].apply(next_working_day)
dates_clean = pd.to_datetime(tweets_clean['Datetime'].dt.date)
tweets_clean.index = dates_clean
# create weights index with business day frequency, with last and first dates of tweets
dt_weights_index = pd.date_range(start=dates_clean.iloc[-1],
                                 end=dates_clean.iloc[0],
                                 freq = 'B')

# create weights dataframe with 0s, the index we created and tickers from our dataframe
df_weights = pd.DataFrame(data = 0,
                          index = dt_weights_index,
                          columns = tickers_new['ticker'])

# reindex returns dataframe because of missing dates (holidays etc)
df_returns_r = df_returns.reindex(df_weights.index, fill_value = 0)

# add a multiindex - dates are not unique (multiple tweets on the same day)
tweets_clean = tweets_clean.set_index(pd.RangeIndex(len(tweets_clean))
                                      ,append= True)

for i in tweets_clean.index:
    tickers = tweets_clean.loc[i,'assign']
    if abs(tweets_clean.loc[i, 'sent']) < p['sent_min']:
        continue
    if p['weighting'] == 'direction':
        sentiment = np.sign(tweets_clean.loc[i, 'sent']) * -1 # multiply by -1: Cramer inverse
    else:
        sentiment = tweets_clean.loc[i, 'sent'] * -1
    for tick in tickers:
        if tick not in list(df_weights.columns):
            print(f"{tick} not in S&P500 and was not manually added. Weight assignment is "
                  f"skipped - you might add the ticker manually and re-run the script")
        else:
            date = i[0] + pd.offsets.BDay(p['trading_lag'])

            if p['take_profit']:
                tp_holding_period = tp_date(df_returns_r.loc[date:,tick],
                                            np.sign(sentiment),
                                            p['take_profit_pct'])

                if tp_holding_period < p['holding_period'] and tp_holding_period !=0:
                    date_till = date + pd.offsets.BDay(tp_holding_period)
                else:
                    date_till = date + pd.offsets.BDay(p['holding_period'])
            else:
                date_till = date + pd.offsets.BDay(p['holding_period'])
            if p['allow_cumulating']:
                df_weights.loc[date:date_till, tick] += sentiment
            elif p['allow_overwrite']:
                df_weights.loc[date:date_till, tick] = sentiment
            elif (df_weights.loc[date:date_till, tick] == 0 ).all():
                df_weights.loc[date:date_till, tick] = sentiment



# scale weights with predefined function
df_weights_scaled = weights_scaler(df_weights.fillna(0), **p['caps'])
# multiple weights with single-stock returns
df_returns_portfolio = df_weights_scaled.mul(df_returns_r).fillna(0)
# sum returns of single stocks to get series of portfolio returns
df_port = pd.DataFrame(df_returns_portfolio.sum(axis=1), columns = ['Jim Cramer'])
# add benchmark S&P500
df_port['S&P500'] = df_returns_r['SPY'].fillna(0)
# use our portfolio analytics class
summary = PortfolioAnalysis(df_port['Jim Cramer'],
                  benchmark = df_port['S&P500'],
                  ann_factor = 252)


summary.navs.plot()
plt.title('NAVs of Jim Cramer inverse strategy and S&P 500')
plt.show()
df_summary = summary.summary_with_benchmark()
summary.navs
