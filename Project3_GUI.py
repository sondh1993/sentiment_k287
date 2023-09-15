import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
plt.style.use('ggplot')
import underthesea
from underthesea import word_tokenize, pos_tag, sent_tokenize
import re
import streamlit as st

st.set_page_config(page_title="Sentiment Analysis", page_icon="📈")
data = pd.read_csv('Sendo_reviews.csv')
# Upload file
uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='latin-1')
    data.to_csv("spam_new.csv", index = False)

# Đọc file trích xuất
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

################################
def find_words(document, list_of_words):
    document_lower = document.lower()
    word_count = 0
    word_list = []

    for word in list_of_words:
        if word in document_lower:
            print(word)
            word_count += document_lower.count(word)
            word_list.append(word)

    return word_count, word_list

def count_value_text(row, find_text):
    row = row.lower()
    count = sum(1 for text in find_text if text in row)
    return count


def pre_text(text, emoji_dict, teen_dict, wrong_lst):
    doc = text.lower()
    doc = doc.replace("'",'')
    doc = re.sub(r'\.+', ".",doc)
    new_sentiment = ''
    for sentence in sent_tokenize(doc):
        # Cover emojicon
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        # conver teencode
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        # dell 
        pattern = r'(?i)\b[b-záâàăắằẵặấầẫẩậbcdđeéèẻẽẹêếềểễệfghíìỉĩịjklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvwxyýỳỷỹỵz]+\b'
        sentence = ' '.join(re.findall(pattern, sentence))
        # del wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentiment = new_sentiment+ sentence + '. '
    doc = new_sentiment
    doc = re.sub(r'\s+', ' ', doc).strip()
    return doc
# Chuẩn hóa unicode tiếng việt
def loaddicchar(txt):
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)
def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, chả...
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()
################################
def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = re.sub(r'\s+', ' ', new_document).strip()
    return new_document
################################
def remove_stop(text, stopwords):
    doc = ' '.join('' if word in stopwords else word for word in text.split())
    doc = re.sub(r'\s+', ' ', doc).strip()
    return doc
################################
def find_words(document, list_of_words):
    document_lower = document.lower()
    word_count = 0
    word_list = []

    for word in list_of_words:
        if word in document_lower:
            print(word)
            word_count += document_lower.count(word)
            word_list.append(word)

    return word_count, word_list
def count_value_text(row, find_text):
    row = row.lower()
    count = sum(1 for text in find_text if text in row)
    return count

# file có sẵn clean text
emoji_dict = file_to_dict(r'emojicon.txt')
teen_dict = file_to_dict(r'teencode.txt')
envi_dict = file_to_dict(r'english-vnmese.txt')
wrong_lst = file_to_list(r'wrong-word.txt')
stop_lst = file_to_list(r'vietnamese-stopwords.txt')
# load file -> create new columns from the values enumeratedsentences
positive_emoji_dict = file_to_list(r'positive_emoji.txt')
negative_emoji_dict = file_to_list(r'negative_emoji.txt')

positive_words_dict = file_to_list(r'positive_words.txt')
negative_words_dict = file_to_list(r'negative_words.txt')

def step_2(df_sub):
    df_sub['words'] = df_sub['content'].apply(lambda x: pre_text(str(x), emoji_dict, teen_dict, wrong_lst))
    df_sub['words'] = df_sub['words'].apply(lambda x: loaddicchar(str(x)))
    df_sub['words'] = df_sub['words'].apply(lambda x: process_postag_thesea(str(x)))
    df_sub['words'] = df_sub['words'].apply(lambda x: remove_stop(str(x), stop_lst))
    df_sub['positive_words_count'] = df_sub['words'].apply(lambda x: count_value_text(str(x), positive_words_dict))
    df_sub['positive_emoji_count'] = df_sub['words'].apply(lambda x: count_value_text(str(x), positive_emoji_dict))
    df_sub['negative_words_count'] = df_sub['words'].apply(lambda x: count_value_text(str(x), negative_words_dict))
    df_sub['negative_emoji_count'] = df_sub['words'].apply(lambda x: count_value_text(str(x), negative_emoji_dict))
    df_sub['words_length'] = df_sub['words'].apply(lambda x: len(str(x).split()))
    # Thêm 2 cột sentiment và đánh giá lại rating
    df_sub['positive'] = df_sub['positive_emoji_count'] + df_sub['positive_words_count']
    df_sub['negative'] = df_sub['negative_emoji_count'] + df_sub['negative_words_count']
    # thêm cột mới rating_new
    df_sub['rating_new'] = np.where(df_sub['positive'] > df_sub['negative'], df_sub['rating']+1, df_sub['rating'] -1 )
    # Tạo cột mới sentiment kết hơp từ rating 
    df_sub['sentiment'] = df_sub['rating_new'].apply(lambda x: '2' if x >= 4 else '0' if x <= 2 else '1')
    # Xóa dữ liệu thiếu
    df_sub = df_sub.dropna()
    df_sub = df_sub.drop(['content','rating'], axis=1)
    return df_sub



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
    st.header("Exploratory Data Analysis")
    st.dataframe(data.sample(20))
    # cột content cần được xử lý
    df_sub = data[['content','rating']].copy()
    df_sub= step_2(df_sub)
    st.markdown("Biểu đồ đếm số lượng sentiment")
    # Tạo biểu đồ đếm số lượng các giá trị trong cột "sentiment"
    ax = sns.countplot(x=df_sub['sentiment'], order=df_sub['sentiment'].value_counts(ascending=True).index)

    # Thêm nhãn số lượng lên trên các cột
    abs_value = df_sub['sentiment'].value_counts(ascending=True).values
    ax.bar_label(ax.containers[0], labels=abs_value)

    # Hiển thị biểu đồ
    st.pyplot(plt)
