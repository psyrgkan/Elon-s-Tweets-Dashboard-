import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from PIL import Image
import seaborn as sns
from datetime import datetime, date

st.sidebar.markdown("EMO Elon")
st.title("Elon is sentimental")
colors = ["#14171A", "#657786", "#1DA1F2"]
sns.set_palette(sns.color_palette(colors))

#Load Data
elon1 = pd.read_csv(r'../data/elon/ElonTweets(Sentiment).csv')
elon2 = pd.read_csv(r'../data/elon/ElonTweets(Sentiment) 10-28-22.csv')
elon3 = pd.read_csv(r'../data/elon/ElonTweets(Sentiment)_11-9-22.csv')
elon = pd.concat([elon1, elon2, elon3], ignore_index=True)
del elon['Unnamed: 0']
del elon['Unnamed: 0.1']
del elon['Unnamed: 0.2']
del elon['verified']
elon.drop_duplicates(subset='Tweet Id', inplace=True)

elon['Date'] = elon[['Date']].applymap(lambda datestr: pd.to_datetime(datestr))

elon['pos_neg_neu'] = elon[['sentiment']].applymap(lambda str: str.split(",")[0])
elon[['pos_neg_neu']] = elon[['pos_neg_neu']].applymap(lambda str: str.replace("'", ""))
elon[['pos_neg_neu']] = elon[['pos_neg_neu']].applymap(lambda str: str.replace("[", ""))
elon_use = elon
# Description
st.sidebar.markdown("In this page we calculate the effect of Elon's sentiment on tweets' performance and the effect of the tweets on his sentiment")

genre = st.sidebar.radio(
    "What effect do you want to see",
    ('Sentiment on performance', 'Tweets on Elon'))

if genre == 'Sentiment on performance':
    col1, col2 = st.columns(2)

    with col1:
        min = elon['Date'].min()
        max = elon['Date'].max()
        start_date_nn = st.date_input('Start date', value=date(2020, 10, 10), min_value=min, max_value=max)
        start_date = np.datetime64(start_date_nn)
        end_date = np.datetime64(st.date_input('End date', max, min_value=start_date_nn, max_value=max))

        if start_date > end_date:
            st.error('Error: End date must fall after start date.')

        elon_use = elon.loc[(elon['Date'] > start_date) & (elon['Date'] < end_date)]

    with col2:
        likes = elon_use.pivot_table(index='pos_neg_neu', aggfunc='mean')[['like count']]
        retweets = elon_use.pivot_table(index='pos_neg_neu', aggfunc='mean')[['retweet count']]
        replies = elon_use.pivot_table(index='pos_neg_neu', aggfunc='mean')[['reply count']]
        lrr = likes.join([retweets, replies])
        lrr.columns = ['Avg Likes', 'Avg Retweets', 'Avg Replies']
        st.bar_chart(lrr)


elif genre == 'Tweets on Elon':
    st.write("eg")