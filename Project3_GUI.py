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
    st.markdown("Biểu đồ đếm số lượng sentiment")
    # Tạo biểu đồ đếm số lượng các giá trị trong cột "sentiment"
    ax = sns.countplot(x=df_sub['sentiment'], order=df_sub['sentiment'].value_counts(ascending=True).index)
    # Thêm nhãn số lượng lên trên các cột
    abs_value = df_sub['sentiment'].value_counts(ascending=True).values
    ax.bar_label(ax.containers[0], labels=abs_value)
    # Hiển thị biểu đồ
    st.pyplot(plt)

    st.markdown("Biểu đồ phần trăm sentiment")
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
