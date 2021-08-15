import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize,sent_tokenize
import re,string,unicodedata
from sklearn.ensemble import RandomForestClassifier
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

class Input(BaseModel):
    text: str

app = FastAPI()

stop_words = stopwords.words('english')
def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    text=text.lower()
    return text

# Tokenizing the training and the test set
def token(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return text

def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
def lemmitize(text):
    lemm=WordNetLemmatizer()
    text= ' '.join([lemm.lemmatize(word) for word in text.split()])
    return text

loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
vectorizer = pickle.load(open('TFIDF.sav', 'rb'))


def pipeline(text):
    message=str(input)
    text=remove_stopwords(text)
    text=remove_special_characters(text)
    text=token(text)
    text= simple_stemmer(text)
    text=lemmitize(text)
    vect_message=vectorizer.transform([text])
    print(vect_message)
    return vect_message

@app.post("/predict")
def predict(item:Input):
    clean_text=pipeline(item.text)
    predict=loaded_model.predict(clean_text)
    print(predict)
    if predict[0]==0:
        return "HQ"
    if predict[0]==1:
        return "Lq_close"
    else:
        return "Lq_edit"
        