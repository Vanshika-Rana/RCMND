import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

drama_df = pd.read_csv("./kdrama.csv")
tfv = TfidfVectorizer(min_df=3, max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3),
                      stop_words='english')
drama_df['Synopsis'] = drama_df['Synopsis'].fillna('')
tfv_matrix = tfv.fit_transform(drama_df['Synopsis'])

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
ind = pd.Series(drama_df.index, index=drama_df['Name']).drop_duplicates()


def rcmnd(title, sug, sig=sig):
    idx = ind[title]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:sug+1]
    drama_idx = [i[0] for i in sig_scores]
    return drama_df['Name'].iloc[drama_idx]


st.title('RCMND')
st.image('./header.jfif')
st.write(" 'RCMND' is here to help you find your next K-Drama to binge watch from the Top 100 K-Drama List. It will recommend you dramas similar to the one you ABSOLUTELY! love <3")
drama_title = st.selectbox('Enter Your Favourite Drama:', drama_df['Name'])
sug = st.slider('Number of Suggestions: ', 1, 20)
recommendedList = rcmnd(drama_title,sug)
finalList = recommendedList.tolist()

st.subheader("Recommended Dramas For You")
for index in range(0, len(finalList)):
    st.write("{0}. {1}".format(index+1, finalList[index]))
