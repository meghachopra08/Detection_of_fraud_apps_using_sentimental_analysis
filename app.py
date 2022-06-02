import pickle
from google_play_scraper import app
import nltk
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import streamlit as st
import numpy as np

model = pickle.load(open("model.pkl",'rb'))
st.title("Google Play Store - Real or Fake")
def preprocess(text):  
    text = text.translate(string.punctuation)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text

link = st.text_input("Enter Play Store App Link")
predict = st.button("Predict")

if predict:
# link="https://play.google.com/store/apps/details?id=tech.plink.PlinkApp"
    findId=link.find('id=')

    url=link[findId+3:]
    file = open("c.txt", "w",encoding='utf-8')
    file.write(str(app(
        url,
        lang='en', # defaults to 'en'
        country='us' # defaults to 'us'
    )))
    file.close()

    myfile=[]
    with open("c.txt",encoding='utf8') as mydata:
        for data in mydata:
            myfile.append(data)
    start=myfile[0].find('comments')
    end=myfile[0].find('editorsChoice')
    c=data[start:end]
    c = preprocess(c)
    c= c.lower() 
    c =  re.sub('[^a-zA-z0-9\s]','',c) 
    c= c.replace('rt','')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(c)

    sequences = tokenizer.texts_to_sequences([c])
    data = pad_sequences(sequences,maxlen = 13661)
    result=model.predict_proba(data)

    class_max = np.argmax(result)
    df = pd.DataFrame({
        'Result':['Real' if class_max==1 else 'Fake'],
        'Score': [round(result[0][class_max],3)]
    })
    # df.reset_index(drop=True, inplace=True)
    st.write(df)
