import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
plt.style.use('ggplot')
import re
import streamlit as st
from collections import Counter
import sys

st.set_page_config(page_title="Sentiment Analysis", page_icon="📈")
data = pd.read_csv('Sendo_reviews.csv')
# Upload file
uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='latin-1')
    data.to_csv("project3.csv", index = False)


# GUI
menu = ["Business Objective", "EDA", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
if choice =="Business Objective":
    st.title("Data Project 3")
    st.write(""" Xây dựng mô hình dự đoán:  
    - giúp người bán hàng có thể biết được những phản hồi nhanh chóng của khách hàng về sản phẩm hay dịch vụ của họ (tích cực, tiêu cực hay trung tính), 
    - điều này giúp cho người bán biết được tình hình kinh doanh, hiểu được ý kiến của khách hàng từ đó giúp họ cải thiện hơn trong dịch vụ, sản phẩm """)
    st.markdown("Dữ Liệu :")
    
    st.dataframe(data.head())
    
elif choice== "EDA":
    df_sub = pd.read_csv('project3_clean.csv')
    st.header("Exploratory Data Analysis")
    st.write("After data preprocessing:")
    st.dataframe(df_sub.sample(20))
    st.title("Biểu đồ đếm số lượng sentiment")
    # Tạo biểu đồ đếm số lượng các giá trị trong cột "sentiment"
    ax = sns.countplot(x=df_sub['sentiment'], order=df_sub['sentiment'].value_counts(ascending=True).index)
    # Thêm nhãn số lượng lên trên các cột
    abs_value = df_sub['sentiment'].value_counts(ascending=True).values
    ax.bar_label(ax.containers[0], labels=abs_value)
    # Hiển thị biểu đồ
    st.pyplot(plt)

    st.title("Biểu đồ phần trăm sentiment")
    sentiment_counts = df_sub['sentiment'].value_counts()
    labels = sentiment_counts.index
    sizes = (sentiment_counts / sentiment_counts.sum()) * 100
    # Vẽ biểu đồ tròn
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    # Vẽ vòng tròn trong để tạo thành biểu đồ tròn
    circle = plt.Circle((0, 0), 0.7, fc='white')
    ax.add_artist(circle)
    # Đảm bảo biểu đồ tròn không bị méo
    ax.axis('equal')
    # Hiển thị biểu đồ
    st.pyplot(fig)
    
    # Biểu đồ tương quan (Correlation matrix)
    st.title("Top 10 customer with the most review")
    # Group data by customer_id and count reviews
    review_counts = data['customer_id'].value_counts()

    # Get the top 10 customers with the most reviews
    top_customers = review_counts.head(10)
    st.bar_chart(top_customers)

    df_positive = df_sub[df_sub['sentiment'] == 'positive']
    df_negative = df_sub[df_sub['sentiment'] == 'negative']
    positive_text = ' '.join(df_positive.dropna()['content'])
    negative_text = ' '.join(df_negative.dropna()['content'])
    wordcloud_positive = WordCloud(background_color='white').generate(positive_text)
    wordcloud_negative = WordCloud(background_color='white').generate(negative_text)
        # Get word frequencies from positive and negative word clouds
    word_freq_positive = wordcloud_positive.words_
    word_freq_negative = wordcloud_negative.words_

    # Sort word frequencies in descending order
    sorted_word_freq_positive = sorted(word_freq_positive.items(), key=lambda x: x[1], reverse=True)
    sorted_word_freq_negative = sorted(word_freq_negative.items(), key=lambda x: x[1], reverse=True)

    # Get the top 10 words and their frequencies for positive sentiment
    top_words_positive = sorted_word_freq_positive[:10]
    words_positive = [word[0] for word in top_words_positive]
    frequencies_positive = [word[1] for word in top_words_positive]

    # Get the top 10 words and their frequencies for negative sentiment
    top_words_negative = sorted_word_freq_negative[:10]
    words_negative = [word[0] for word in top_words_negative]
    frequencies_negative = [word[1] for word in top_words_negative]
    # Display word cloud for positive text
    st.subheader('Word Cloud - Positive Sentiment')
    st.image(wordcloud_positive.to_array(), use_column_width=True)

    # Display word cloud for negative text
    st.subheader('Word Cloud - Negative Sentiment')
    st.image(wordcloud_negative.to_array(), use_column_width=True)

    # Create bar plots for word frequencies in positive and negative sentiment
    st.subheader('Top 10 Words - Positive Sentiment')
    st.bar_chart(frequencies_positive, labels=words_positive)

    st.subheader('Top 10 Words - Negative Sentiment')
    st.bar_chart(frequencies_negative, labels=words_negative)
    # Display word cloud for positive text
    st.subheader('Word Cloud - Positive Sentiment')
    st.image('wordcloud_positive.png', use_column_width=True)

    # Display word cloud for negative text
    st.subheader('Word Cloud - Negative Sentiment')
    st.image('wordcloud_negative.png', use_column_width=True)