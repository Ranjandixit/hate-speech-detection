import re
import os
import string
import streamlit as st
from db import Message, Predictor
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


st.title("Hate Speech Detection")

def tokenize(s):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    out = regex.sub(' ', s).split()
    return out

def fit_logistic(x, y):
    y = y.values
    model = LogisticRegression(C=4, dual=True)
    return model.fit(x, y)

@st.cache
def load_dataset(path='archive/labeled_data.csv'):
    df = pd.read_csv(path)
    df.drop(['count','hate_speech','offensive_language','neither'],axis=1,inplace=True)
    return df

def load_vectorizer():
    with open(r"models\tfidf_vectorizer_train.pkl",'rb') as f:
        return pickle.load(f)
def load_hate_model():
    with open(r"models\logistic_hate_speech.pkl",'rb') as f:
        return pickle.load(f)
def load_language_model():
    with open(r"models\logistic_offensive_language.pkl",'rb') as f:
        return pickle.load(f)
def load_class_type():
    with open(r"models\logistic_class.pkl",'rb') as f:
        return pickle.load(f)

def opendb():
    engine = create_engine('sqlite:///db.sqlite3') # connect
    Session =  sessionmaker(bind=engine)
    return Session()

def save_message(text,name='ranjan'):
    try:
        db = opendb()
        msg = Message(text=text,uploader=name)
        db.add(msg)
        db.commit()
        db.close()
        return True
    except Exception as e:
        st.write("database error:",e)
        return False

def delete_message(msg_id):
    try:
        db = opendb()
        msg = db.query(Message).filter(Message.id==msg_id).delete()
        db.commit()
        db.close()
        return True
    except Exception as e:
        st.write("database error:",e)
        return False

cls_names =[" hate speech","offensive language","neither"]


choice = st.sidebar.selectbox("select option",[
    'About Project',
    'Detect HATE SPEECH',
    'Manage data',
    'View History',
    'View Dataset',
    ])

hate = load_hate_model()
tfidf = load_vectorizer()
language = load_language_model()
classtype = load_class_type()

if choice == 'About Project':
    # jo krna ho krna
    pass
if choice == 'Detect HATE SPEECH':
    
    st.success("models loaded")
    uploader = st.text_input("person name")
    comment = st.text_area('comment or tweet or post content')
    if st.button("predict speech"):
        if comment and uploader:
            x = tfidf.transform(np.array([comment]))
            h = hate.predict(x)[0]
            ol = language.predict(x)[0]
            c = classtype.predict(x)[0]
            st.header("person:",uploader)
            if c == 0:
                st.title("The text is a hate speech")
                st.info(f"hate speech rating:{h}")
            elif c == 1:
                st.title("The text is not hate speech but is offensive")
                st.info(f"hate speech rating:{ol}")
            else:
                st.title("not a hate speech or offensive comment")
            save_message(comment,name=uploader)
            st.success("saved in database")
        else:
            st.error("please fill valid values")

if choice == 'View History':
    db = opendb()
    contents = db.query(Message).all()
    db.close()
    comment = st.selectbox("select a message",contents)
    if st.button("predict"):
        x = tfidf.transform(np.array([comment.text]))
        h = hate.predict(x)[0]
        ol = language.predict(x)[0]
        c = classtype.predict(x)[0]
        st.header(f"person:{comment.uploader}")
        st.subheader(f"comment:{comment}")
        if c == 0:
            st.title("The text is a hate speech")
            st.info(f"hate speech rating:{h}")
        elif c == 1:
            st.title("The text is not hate speech but is offensive")
            st.info(f"hate speech rating:{ol}")
        else:
            st.title("not a hate speech or offensive comment")

if choice == 'View Dataset':
    df = load_dataset()
    viewsize = st.selectbox('view data',['top 5 row','bottom 5 rows','all'])
    if viewsize == 'top 5 row':
        st.write(df.head())
    if viewsize == 'bottom 5 rows':
        st.write(df.tail())
    if viewsize == 'all':
        st.write(df)

if choice ==  'Manage data':
    db = opendb()
    results = db.query(Message).all()
    db.close()
    msg = st.radio('select image',results)
    if msg and st.button('delete msg'):
        if delete_message(msg.id):
            st.success("message deleted")
        else:
            st.error("message could not be deleted")