import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
# from underthesea import word_tokenize, pos_tag, sent_tokenize
import re
import streamlit as st
# đọc file 
df = pd.read_csv('Sendo_reviews.csv')

# GUI
st.title("Data Science Project")
st.write("## Ham vs Spam")
# Upload file
uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='latin-1')
    data.to_csv("spam_new.csv", index = False)


# GUI
menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ######### Xây dựng mô hình dự đoán:  
- giúp người bán hàng có thể biết được những phản hồi nhanh chóng của khách hàng về sản phẩm hay dịch vụ của họ (tích cực, tiêu cực hay trung tính), 
- điều này giúp cho người bán biết được tình hình kinh doanh, hiểu được ý kiến của khách hàng từ đó giúp họ cải thiện hơn trong dịch vụ, sản phẩm

<img src="logo.jpg" alt="image_description" width="300" height="200"> """)  


elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data")
    st.dataframe(data[['v2', 'v1']].head(3))
    st.dataframe(data[['v2', 'v1']].tail(3))  
    st.write("##### 2. Visualize Ham and Spam")
    fig1 = sns.countplot(data=data[['v1']], x='v1')    
    st.pyplot(fig1.figure)

    st.write("##### 3. Build model...")
    st.write("##### 4. Evaluation")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    st.code("Accuracy:"+str(round(acc,2)))
    st.write("###### Confusion matrix:")
    st.code(cm)
    st.write("###### Classification report:")
    st.code(cr)
    st.code("Roc AUC score:" + str(round(roc,2)))

    # calculate roc curve
    st.write("###### ROC curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    fig, ax = plt.subplots()       
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.plot(fpr, tpr, marker='.')
    st.pyplot(fig)

    st.write("##### 5. Summary: This model is good enough for Ham vs Spam classification.")
