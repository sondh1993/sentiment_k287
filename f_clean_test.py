from underthesea import word_tokenize, pos_tag, sent_tokenize
import re
from underthesea import word_tokenize, pos_tag, sent_tokenize
import numpy as np
# Count emojis positive and negative
negative_emojis = [
    "😞", "😔", "🙁", "☹️", "😕",
    "😢", "😭", "😖", "😣", "😩",
    "😠", "😡", "🤬", "😤", "😰",
    "😨", "😱", "😪", "😓", "🥺",
    "😒", "🙄", "😑", "😬", "😶",
    "🤯", "😳", "🤢", "🤮", "🤕",
    "🥴", "🤔", "😷", "🙅‍♂️", "🙅‍♀️",
    "🙆‍♂️", "🙆‍♀️", "🙇‍♂️", "🙇‍♀️", "🤦‍♂️",
    "🤦‍♀️", "🤷‍♂️", "🤷‍♀️", "🤢", "🤧",
    "🤨", "🤫", "👎", "👊", "✊", "🤛", "🤜",
    "🤚", "🖕"
]

positive_emojis = [
    "😄", "😃", "😀", "😁", "😆",
    "😅", "🤣", "😂", "🙂", "🙃",
    "😉", "😊", "😇", "🥰", "😍",
    "🤩", "😘", "😗", "😚", "😙",
    "😋", "😛", "😜", "🤪", "😝",
    "🤗", "🤭", "🥳", "😌", "😎",
    "🤓", "🧐", "👍", "🤝", "🙌", "👏", "👋",
    "🤙", "✋", "🖐️", "👌", "🤞",
    "✌️", "🤟", "👈", "👉", "👆",
    "👇", "☝️"
]

positive_words = [
    "thích", "tốt", "xuất sắc", "tuyệt vời", "tuyệt hảo", "đẹp", "ổn"
    "hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "thú vị", "nhanh"
    "tiện lợi", "dễ sử dụng", "hiệu quả", "ấn tượng",
    "nổi bật", "tận hưởng", "tốn ít thời gian", "thân thiện", "hấp dẫn",
    "gợi cảm", "tươi mới", "lạ mắt", "cao cấp", "độc đáo",
    "hợp khẩu vị", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp",
    "hấp dẫn", "an tâm", "không thể cưỡng lại", "thỏa mãn", "thúc đẩy",
    "cảm động", "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội",
    "sáng tạo", "quý báu", "phù hợp", "tận tâm",
    "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận",
    "vui vẻ", "sáng sủa", "hào hứng", "đam mê", "vừa vặn", "đáng tiền"
]
negative_words = [
    "kém", "tệ", "đau", "xấu",
    "buồn", "rối", "thô", "lâu"
    "tối", "chán", "ít", "mờ", "mỏng",
    "lỏng lẻo", "khó", "cùi", "yếu",
    "kém chất lượng", "không thích", "không thú vị", "không ổn"
    "không hợp", "không đáng", "không chuyên nghiệp",
    "không phản hồi", "không an toàn", "không phù hợp", "không thân thiện", "không linh hoạt", "không đáng",
    "không ấn tượng", "không tốt", "chậm", "khó khăn", "phức tạp",
    "khó hiểu", "khó chịu", "gây khó dễ", "rườm rà", "khó truy cập",
    "thất bại", "tồi tệ", "khó xử", "không thể chấp nhận", "tồi tệ","không rõ ràng",
    "không chắc chắn", "rối rắm", "không tiện lợi", "không đáng", "chưa đẹp", "không đẹp"
]
def save_emojis_to_file(emojis, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for emoji in emojis:
            file.write(emoji + '\n')

def save_text_to_file(texts, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for text in texts:
            file.write(text + '\n')

            
save_emojis_to_file(negative_emojis, 'negative_emojis_list.txt')
save_emojis_to_file(positive_emojis, 'positive_emojis_list.txt')
save_text_to_file(negative_words, 'negative_words_list.txt')
save_text_to_file(positive_words, 'positive_words_list.txt')

def file_to_dict(path):
    file = open(path, 'r', encoding = 'utf-8')
    lst = file.read().split('\n')
    dict = {}
    for line in lst:
        key, value = line.split('\t')
        dict[key] = str(value)
    file.close()
    return dict
def file_to_list(path):
    file = open(path, 'r', encoding = 'utf-8')
    lst = file.read().split('\n')
    file.close
    return lst

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
wrong_lst = file_to_list(r'wrong-word.txt')
stop_lst = file_to_list(r'vietnamese-stopwords.txt')
######

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
################################
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

################################
def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, chả...
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
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
def translate_text(text):
    words = text.split()
    translated_words = [envi_dict.get(word.lower(), word) for word in words]
    return ' '.join(translated_words)
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
###############################################################
def count_value_text(row, find_text):
    row = row.lower()
    count = sum(1 for text in find_text if text in row)
    return count

def step_clean(df_sub):
    df_sub['words'] = df_sub['content'].apply(lambda x: pre_text(str(x), emoji_dict, teen_dict, wrong_lst))
    df_sub['words'] = df_sub['words'].apply(lambda x: loaddicchar(str(x)))
    df_sub['words'] = df_sub['words'].apply(lambda x: translate_text(x))
    # Tính toán số lượng từ và emoji mang tính cảm xúc
    df_sub['positive_words_count'] = df_sub['words'].apply(lambda x: count_value_text(str(x), positive_words_dict))
    df_sub['positive_emoji_count'] = df_sub['words'].apply(lambda x: count_value_text(str(x), positive_emoji_dict))
    df_sub['negative_words_count'] = df_sub['words'].apply(lambda x: count_value_text(str(x), negative_words_dict))
    df_sub['negative_emoji_count'] = df_sub['words'].apply(lambda x: count_value_text(str(x), negative_emoji_dict))
    
    df_sub['words'] = df_sub['words'].apply(lambda x: word_tokenize(x))
    df_sub['words'] = df_sub['words'].apply(lambda x: process_postag_thesea(str(x)))
    df_sub['words'] = df_sub['words'].apply(lambda x: remove_stop(str(x), stop_lst))
   
    df_sub['words_length'] = df_sub['words'].apply(lambda x: len(str(x).split()))
    # Thêm 2 cột sentiment và đánh giá lại rating
    df_sub['positive'] = df_sub['positive_emoji_count'] + df_sub['positive_words_count']
    df_sub['negative'] = df_sub['negative_emoji_count'] + df_sub['negative_words_count']
    # thêm cột mới rating_new
    df_sub['rating_new'] = np.where(df_sub['positive'] > df_sub['negative'], df_sub['rating']+1, df_sub['rating'] -1 )
    # Tạo cột mới sentiment kết hơp từ rating 
    df_sub['sentiment'] = df_sub['rating_new'].apply(lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutal')
    # Xóa dữ liệu thiếu
    df_sub_new = df_sub.dropna()
    return df_sub_new

