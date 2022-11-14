import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from PIL import Image
import seaborn as sns
from datetime import datetime, date

st.sidebar.markdown("Dashboard")
st.title("Welcome to Elon's Tweets!")
colors = ["#14171A", "#657786", "#1DA1F2"]
sns.set_palette(sns.color_palette(colors))

#Load Data
@st.cache
def load_data():

    elon1 = pd.read_csv(r'./elon/ElonTweets(Sentiment).csv')
    elon2 = pd.read_csv(r'./elon/ElonTweets(Sentiment) 10-28-22.csv')
    elon3 = pd.read_csv(r'./elon/ElonTweets(Sentiment)_11-9-22.csv')
    elon = pd.concat([elon1, elon2, elon3], ignore_index=True)
    del elon['Unnamed: 0']
    del elon['Unnamed: 0.1']
    del elon['Unnamed: 0.2']
    del elon['verified']
    # st.write(elon.head())
    elon.drop_duplicates(subset='Tweet Id', inplace=True)

    elon['Date'] = elon[['Date']].applymap(lambda datestr: pd.to_datetime(datestr))

    elon['pos_neg_neu'] = elon[['sentiment']].applymap(lambda str: str.split(",")[0])
    elon[['pos_neg_neu']] = elon[['pos_neg_neu']].applymap(lambda str: str.replace("'", ""))
    elon[['pos_neg_neu']] = elon[['pos_neg_neu']].applymap(lambda str: str.replace("[", ""))
    return elon

elon = load_data()

#Datetime Selector
st.sidebar.header("Elon's Tweets")
st.sidebar.markdown("In this dashboard we can explore a couple of statistics regarding Elon's tweets for the past ten years or so.  " 
                        "\n Below please select the dates you would like analyzed")
min = elon['Date'].min()
max = elon['Date'].max()
col1, col2 = st.columns(2)
with col1:
    start_date_nn = st.sidebar.date_input('Start date', value=date(2020, 10, 10), min_value=min, max_value=max)
    start_date = np.datetime64(start_date_nn)
with col2:
    end_date = np.datetime64(st.sidebar.date_input('End date', max, min_value=start_date_nn, max_value=max))

if start_date < end_date:
    st.success('Start date: `%s` \t \t End date:`%s`' % (start_date, end_date))
else:
    st.error('Error: End date must fall after start date.')

elon_use = elon.loc[(elon['Date'] > start_date) & (elon['Date'] < end_date)]

options = st.multiselect(
    'What stats are you most interested in:',
    ['Followers', 'Most frequent words in Tweets', 'Most frequent mentions in Tweets', 'Likes/Replies/Retweets', 'Sentiment'],
    ['Likes/Replies/Retweets', 'Sentiment'])

# st.write('You selected:', options)

#Plot follower count over time
with st.expander("Click here to view Follower line chart"):
    # fol = plt.figure()
    # sns.lineplot(data=elon_use, x='Date', y='Follower Count', color="#1DA1F2")
    # plt.xticks(rotation=60)
    # st.pyplot(fol)
    if 'Followers' in options:
        st.line_chart(data=elon_use, x='Date', y='Follower Count')

#Sentiment counts
with st.expander("Click here to view Sentiment chart"):
    sentiment = elon_use.pivot_table(index='pos_neg_neu', aggfunc='count')[['sentiment']]
    sentiment.reset_index(inplace=True)

    # #Plot count of sentiment
    # bars = plt.figure()
    # sns.barplot(sentiment, y='sentiment', x='pos_neg_neu')
    # plt.xlabel("Sentiment")
    # plt.ylabel("Count")
    # st.pyplot(bars)

    if 'Sentiment' in options:
        st.bar_chart(data=sentiment, x='pos_neg_neu', y='sentiment')

#Like counts
with st.expander("Click here to view Like/Retweet/Reply chart"):
    likes = elon_use.pivot_table(index='Date', aggfunc='sum')[['like count']]
    retweets = elon_use.pivot_table(index='Date', aggfunc='sum')[['retweet count']]
    replies = elon_use.pivot_table(index='Date', aggfunc='sum')[['reply count']]
    lrr = likes.join([retweets, replies])
    lrr.reset_index(inplace=True)

    first_like = int(likes.sort_values('Date').iloc[0:20].mean().values)
    last_like = int(likes.sort_values('Date').iloc[-21:-1].mean().values)

    first_rt = int(retweets.sort_values('Date').iloc[0:20].mean().values)
    last_rt = int(retweets.sort_values('Date').iloc[-21:-1].mean().values)

    first_rp = int(replies.sort_values('Date').iloc[0:20].mean().values)
    last_rp = int(replies.sort_values('Date').iloc[-21:-1].mean().values)

    if 'Likes/Replies/Retweets' in options:
        col1, col2, col3 = st.columns(3)
        col1.metric("Likes", last_like, last_like-first_like)
        col2.metric("Retweets", last_rt, last_rt-first_rt)
        col3.metric("Replies", last_rp, last_rp-first_rp)
        st.line_chart(data=lrr, x='Date', y=['like count', 'retweet count', 'reply count'])
        st.bar_chart(pd.DataFrame(lrr.mean(), columns=["Average"]))

#Create WordClouds
text = " ".join(t for t in elon_use.Text)
text = re.sub(r'http\S+', '', text)
text = re.sub(r'amp\S+', '', text)
text = re.sub(r'@\S+', '', text)
stopwords = set(STOPWORDS)

with st.expander("Click here to view Frequent Tweet words"):

    #First
    if  'Most frequent words in Tweets' in options:
        with st.spinner('Working on it...'):
            mask = np.array(Image.open("./text_bubble.png"))
            word_cloud = WordCloud(stopwords=stopwords, background_color="white", max_words=1000, mask=mask).generate(text)
            wc = plt.figure()
            plt.imshow(word_cloud, interpolation="bilinear")
            plt.axis("off")
            # store to file
            # plt.savefig("wordcloud.png", format="png")
            st.pyplot(wc)

with st.expander('Click here to view Frequent Mentions'):

    #Second
    if 'Most frequent mentions in Tweets' in options:
        with st.spinner('Working on it...'):
            mask = np.array(Image.open("./twitter_logo.png"))
            mentions = WordCloud(stopwords=stopwords, background_color="white", max_words=1000, mask=mask).generate(" ".join(elon_use['mentions']))
            mwc = plt.figure()
            plt.imshow(mentions, interpolation="bilinear")
            plt.axis("off")
            # store to file
            # plt.savefig("wordcloud.png", format="png")
            st.pyplot(mwc)





