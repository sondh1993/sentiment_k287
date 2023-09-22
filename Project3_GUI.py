import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import f_clean_test
import time
import os

def missing_value_analysis(df):
    na_col = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_col].isnull().sum().sort_values(ascending=True)
    ratio_ = (df[na_col].isnull().sum() / df.shape[0]*100).sort_values(ascending=True)
    miss_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=["Missing values", "Ratio"])
    miss_df = pd.DataFrame(miss_df)
    return miss_df

def check_data(df, head=5, tail=5):
    st.write("Rows: {}, Columns: {}".format(df.shape[0],df.shape[1]))
    st.write("Missing Values Analysis")
    miss_df = missing_value_analysis(df)
    st.table(miss_df)
    st.write("Duplicated values:",df.duplicated().sum())
def file_to_dict(path):
    file = open(path, 'r', encoding = 'utf-8')
    lst = file.read().split('\n')
    dict = {}
    for line in lst:
        key, value = line.split('\t')
        dict[key] = str(value)
    return dict
def file_to_list(path):
    file = open(path, 'r', encoding = 'utf-8')
    lst = file.read().split('\n')
    return lst
# Thêm các file dùng để clean dữ liệu
emoji_dict = file_to_dict(r'emojicon.txt')
teen_dict = file_to_dict(r'teencode.txt')
envi_dict = file_to_dict(r'english-vnmese.txt')
wrong_lst = file_to_list(r'wrong-word.txt')
stop_lst = file_to_list(r'vietnamese-stopwords.txt')
positive_emoji_dict = file_to_list(r'positive_emojis_list.txt')
negative_emoji_dict = file_to_list(r'negative_emojis_list.txt')

positive_words_dict = file_to_list(r'positive_words_list.txt')
negative_words_dict = file_to_list(r'negative_words_list.txt')

# file có sẵn clean text
emoji_dict = file_to_dict(r'emojicon.txt')
teen_dict = file_to_dict(r'teencode.txt')
envi_dict = file_to_dict(r'english-vnmese.txt')
df_clean = pd.read_csv("project3_clean.csv")
# Trang 1: Giới thiệu tổng quát về model
def page_intro():
    st.title("Project : Dự đoán Sentiment")
    st.write("Ứng dụng này sử dụng mô hình để dự đoán cảm xúc (sentiment) \n- Dữ liệu được lấy từ sendo, gồm những đánh giá của khách hàng")
    st.image("logo.jpg")
    # Thêm thông tin giới thiệu khác (tuỳ chọn)
    st.markdown("### Giáo viên hướng dẫn :\nKhuất Thùy Dương")
    st.markdown("### Học viên :\nĐặng Huỳnh Sơn")
# Trang 2: Tìm hiểu thuật toán và phân tích dữ liệu
def page_algorithm():
    st.title("Phân tích Khám phá Dữ liệu - EDA")
    st.write("Trang này cung cấp thông tin về thuật toán được sử dụng và phân tích dữ liệu.")
    # Thêm thông tin về thuật toán và phân tích dữ liệu (tuỳ chọn)
    # Lựa chọn tệp tin
    uploaded_file = st.file_uploader("Tải lên tệp tin CSV", type="csv")
    read_data = pd.read_csv("Sendo_reviews.csv")
    # Đọc dữ liệu từ tệp tin hoặc sử dụng dữ liệu mặc định
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        df = read_data
    df = pd.DataFrame(df)
    # Hiển thị dữ liệu
    st.write("Dữ liệu được đọc:")
    st.write(df)
    check_data(df)
    st.write("Một số thông tin từ dữ liệu qua biểu đồ")
    ax = df['rating'].value_counts().sort_index().plot(kind='bar',
                                                       title='Count of Reviews by Rating', figsize=(10,5))
    ax.set_xlabel("Review Rating")
    # Tạo biểu đồ cột
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = df['rating'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Rating')
    ax.set_xlabel("Review Rating")

    # Hiển thị giá trị con số trên biểu đồ
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)
    # Top 10 khách hàng đánh giá nheieuf
    top_customer = df.groupby("customer_id")['rating'].count().sort_values(ascending=False).head(10)
    # Tạo biểu đồ cột bằng Seaborn
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_customer.values, y=top_customer.index, ax=ax)
    # Thiết lập nhãn và tiêu đề
    ax.set_xlabel('Purchase Count')
    ax.set_ylabel('Customer')
    ax.set_title('Top 10 Customers with Highest comment Count')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)
    
    # Nối tất cả các nội dung thành một chuỗi
    text = ' '.join(df['content'].astype(str))

    # Tạo word cloud
    wordcloud = WordCloud(width=900, height=400, background_color='white').generate(text)

    # Hiển thị word cloud trong Streamlit
    st.title('Word Cloud của Content')
    st.image(wordcloud.to_array(), use_column_width=True)

    # Tính phần trăm của các sentiment
    rating_count = df['rating'].value_counts()
    rating_percen = rating_count / rating_count.sum() * 100
    labels = rating_percen.index
    sizes = rating_percen.values

    # Vẽ biểu đồ hình tròn
    sns.set_palette("Set3")  # Đặt màu sắc cho biểu đồ
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Đảm bảo biểu đồ tròn không bị méo

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    st.markdown('''### Qua sơ lược các biểu đồ như trên có thể nhận xét như sau:
- Các content nghiêng về tích cực.
- Nhóm khách hàng đánh giá nhiều lần cho thấy không phải bot.
- Trước khi bắt đầu train dữ liệu dự đoán, tiến hành làm sạch dữ liệu.''')
    
    st.markdown("## Data Cleaning ")
    st.write('Chọn 2 đặc trưng cần thiết để giữ lại content và rating')
    df_sub = df[['content','rating']].copy()
    st.dataframe(df_sub.head(10))
    
    def process_text(df_sub):
        # Xử lý dữ liệu
        df_sub['words'] = df_sub['content'].apply(lambda x: f_clean_test.pre_text(str(x), emoji_dict, teen_dict, wrong_lst))
        df_sub['words'] = df_sub['words'].apply(lambda x: f_clean_test.loaddicchar(str(x)))
        df_sub['words'] = df_sub['words'].apply(lambda x: f_clean_test.translate_text(x))

        # Hiển thị quá trình xử lý dữ liệu
        progress_bar = st.empty()
        progress_text = st.empty()

        # Tính toán số lượng từ và emoji mang tính cảm xúc
        for i in range(len(df_sub)):
            # Cập nhật quá trình xử lý
            progress_bar.progress((i + 1) / len(df_sub))
            progress_text.text(f"Đang xử lý dòng thứ {i + 1}/{len(df_sub)}")

            # Xử lý từng dòng dữ liệu
            row = df_sub.iloc[i]
            row['positive_words_count'] = f_clean_test.count_value_text(str(row['words']), positive_words_dict)
            row['positive_emoji_count'] = f_clean_test.count_value_text(str(row['words']), positive_emoji_dict)
            row['negative_words_count'] = f_clean_test.count_value_text(str(row['words']), negative_words_dict)
            row['negative_emoji_count'] = f_clean_test.count_value_text(str(row['words']), negative_emoji_dict)

            row['words'] = f_clean_test.word_tokenize(str(row['words']))
            row['words'] = f_clean_test.process_postag_thesea(str(row['words']))
            row['words'] = f_clean_test.remove_stop(str(row['words']), stop_lst)

            row['words_length'] = len(str(row['words']).split())
            row['positive'] = row['positive_emoji_count'] + row['positive_words_count']
            row['negative'] = row['negative_emoji_count'] + row['negative_words_count']
            row['rating_new'] = row['rating'] + 1 if row['positive'] > row['negative'] else row['rating'] - 1
            row['sentiment'] = 'positive' if row['rating_new'] >= 4 else 'negative'

            # Tạm ngừng để hiển thị hiệu ứng loading
            time.sleep(0.5)

        # Xóa quá trình xử lý khi hoàn thành
        progress_bar.empty()
        progress_text.empty()

    st.title("Xử lý dữ liệu")
    # Tạo nút "Tải lên dữ liệu"
    # Kiểm tra xem có file đã xử lý sẵn hay không
    processed_file_path = "project3_clean.csv"
    use_processed_file = st.checkbox("Sử dụng file đã xử lý sẵn", value=True)

    if use_processed_file:
        if os.path.exists(processed_file_path):
            # Đọc dữ liệu từ file đã xử lý sẵn
            st.success("Đã hoàn thành xử lý dữ liệu!")
            df_sub = pd.read_csv(processed_file_path)
            st.dataframe(df_sub.sample(10))
        else:
            st.write("File đã xử lý sẵn không tồn tại.")
    else:
        uploaded_file = st.file_uploader("Tải lên file dữ liệu")
        if uploaded_file is not None:
            # Đọc dữ liệu từ file tải lên
            df_sub = pd.read_csv(uploaded_file)

            # Thực hiện xử lý dữ liệu
            process_text(df_sub)

            # Lưu dữ liệu đã xử lý vào file
            df_sub.to_csv(processed_file_path, index=False)
        else:
            st.write("Vui lòng tải lên file dữ liệu.")

    
    
    st.title("Word Cloud and Bar Plot of Text cleaning")
    sentiments = df_sub['sentiment'].unique()
    wordclouds = {}

    for sentiment in sentiments:
        sentiment_text = ' '.join(df_sub[df_sub['sentiment'] == sentiment]['content'])
        wordcloud = WordCloud(background_color='white', collocations=False).generate(sentiment_text)
        wordclouds[sentiment] = wordcloud

    # Hiển thị word clouds trong Streamlit
for sentiment, wordcloud in wordclouds.items():
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud - {sentiment.capitalize()} Sentiment')
    plt.axis('off')
    if sentiment == 'negative':
        st.write('Word Cloud - Negative Sentiment')
        st.pyplot(plt)
        st.write('')  # Thêm một dòng trống giữa các biểu đồ
    elif sentiment == 'positive':
        st.write('Word Cloud - Positive Sentiment')
        st.pyplot(plt)

    # Get word frequencies from word clouds
    word_freqs = {}
    for sentiment, wordcloud in wordclouds.items():
        word_freq = wordcloud.words_
        sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_word_freq[:10]
        words = [word[0] for word in top_words]
        frequencies = [word[1] for word in top_words]
        word_freqs[sentiment] = (words, frequencies)

    # Create bar plots for word frequencies in each sentiment
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, sentiment in enumerate(sentiments):
        words, frequencies = word_freqs[sentiment]
        ax = axes[i // 2, i % 2]
        ax.bar(words, frequencies)
        ax.set_title(f'Top 10 Words - {sentiment.capitalize()} Sentiment')
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', rotation=90)

    # Adjust layout and display the bar plots
    plt.tight_layout()
    st.pyplot(fig)

# Trang 3: Huấn luyện model và kết quả mẫu
def page_training():
    st.title("Huấn luyện model")
    st.write("Trang này cho phép người dùng lựa chọn thuật toán và xem kết quả mẫu sau huấn luyện.")
    # Thêm các lựa chọn thuật toán (tuỳ chọn)
    # Thêm kết quả mẫu dữ liệu thử (tuỳ chọn)

# Trang 4: Người dùng tự nhập
def page_user_input():
    st.title("Người dùng tự nhập")
    st.write("Trang này cho phép người dùng tự nhập văn bản để dự đoán sentiment.")
    user_input = st.text_area("Nhập văn bản của bạn", "")
    # Xử lý văn bản người dùng và gọi mô hình để dự đoán sentiment (tuỳ chọn)

# Thiết lập giao diện ứng dụng Streamlit
def main():
    # Định cấu hình trang
    st.set_page_config(page_title="Dự đoán Sentiment", layout="wide")

    # Hiển thị menu điều hướng
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Chọn trang", ("Giới thiệu", "Phân tích Khám phá Dữ liệu - EDA", "Huấn luyện model", "Người dùng tự nhập"))

    # Hiển thị trang tương ứng dựa trên lựa chọn
    if page == "Giới thiệu":
        page_intro()
    elif page == "Phân tích Khám phá Dữ liệu - EDA":
        page_algorithm()
    elif page == "Huấn luyện model":
        page_training()
    elif page == "Người dùng tự nhập":
        page_user_input()

if __name__ == "__main__":
    main()
    
