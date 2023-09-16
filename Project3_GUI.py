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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score,classification_report, confusion_matrix, precision_score, f1_score

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
    st.subheader("Project: Sentiment Analysis")
    st.write(""" Xây dựng mô hình dự đoán:  
    - giúp người bán hàng có thể biết được những phản hồi nhanh chóng của khách hàng về sản phẩm hay dịch vụ của họ (tích cực, tiêu cực hay trung tính), 
    - điều này giúp cho người bán biết được tình hình kinh doanh, hiểu được ý kiến của khách hàng từ đó giúp họ cải thiện hơn trong dịch vụ, sản phẩm
             
             
             
              """)
    st.subheader("- Giáo viên hướng dẫn: Khuất Thùy Phương\n \
                 - Học viên: Đặng Huỳnh Sơn")
    st.image('logo.jpg')
    
elif choice== "EDA":
    df_sub = pd.read_csv('project3_clean.csv')
    st.title("Exploratory Data Analysis")
    st.subheader("Dữ Liệu được cung cấp:")
    st.dataframe(data)
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
 
    # Display word cloud for positive text
    st.subheader('Word Cloud - Positive Sentiment')
    st.image('Word Cloud - Positive Sentiment.jpg', use_column_width=True)

    # Display word cloud for negative text
    st.subheader('Word Cloud - Negative Sentiment')
    st.image('Word Cloud - Negative Sentiment.png', use_column_width=True)

    # Display top 10 words and their frequencies
    st.subheader('top 10 words and their frequencies')
    st.image('top10word.png', use_column_width=True)
elif choice== "Build Project":
    df_results = pd.read_csv("result_model.csv")
    st.table(df_results)
    # Display top model 
    st.subheader('Result Model RandomForestClassifier')
    st.image('test.png', use_column_width=True)
    
