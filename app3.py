 #pip install pandas scikit-learn
#pip install pandas scikit-learn --no-warn-script-location
#pip install nltk
#pip install arabic_reshaper
#pip install python-bidi

import numpy as np
import pandas as pd
import nltk
import string
import arabic_reshaper
from nltk.corpus import stopwords
from bidi.algorithm import get_display
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
punc = string.punctuation

# قائمة الكلمات الشائعة التي سيتم استبعادها (يمكنك توسيعها)
ARABIC_STOPWORDS = set(stopwords.words('arabic'))

def preprocess_arabic(text):
    """تنظيف النصوص العربية"""
    text = str(text).translate(str.maketrans('','',punc))  # إزالة علامات الترقيم
    words = text.split()  # تقسيم النص إلى كلمات
    words = [word for word in words if word not in ARABIC_STOPWORDS]  # إزالة الكلمات الشائعة
    return " ".join(words)  # إرجاع النص بعد إزالة الكلمات الشائعة

def print_arabic_text(text):
    """لإظهار النصوص العربية بشكل صحيح"""
    reshaped_text = arabic_reshaper.reshape(text)  # تعديل النص ليتناسب مع الاتجاه
    bidi_text = get_display(reshaped_text)  # ضبط النص ليظهر بشكل صحيح في الاتجاه من اليمين لليسار
    print(bidi_text)

# قراءة المستندات من ملف CSV داخل مجلد "data"
file_path = "data/arabic_text_data.csv"  # تأكد من أن الملف موجود داخل مجلد data

df = pd.read_csv(file_path)  # قراءة الملف
documents = np.array(df["text"].dropna())  # استخراج النصوص كمصفوفة بعد إزالة القيم الفارغة

# تحضير البيانات باستخدام TfidfVectorizer
vectorizer = TfidfVectorizer(preprocessor=preprocess_arabic)  # تحديد الدالة المخصصة للمعالجة المبدئية
tfidf_matrix = vectorizer.fit_transform(documents)  # حساب TF-IDF

#  طباعة النتائج لكل مستند
feature_names = vectorizer.get_feature_names_out()  # استخراج أسماء الكلمات (المزايا)
for i, doc_tfidf in enumerate(tfidf_matrix.toarray()):
    print(f"📄 Document {i+1} - TF-IDF values:")
    for j, score in enumerate(doc_tfidf):
        if score > 0:  # عرض الكلمات التي تحتوي على قيم TF-IDF أكبر من الصفر
            print_arabic_text(f"{feature_names[j]}: {score:.4f}")
    print("\n")
