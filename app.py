import os
import streamlit as st
from db import Message, Predictor
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



st.title("Hate Speech Detection")

def load_hate_model(path):
    pass

@st.cache
def load_dataset(path='archive/labeled_data.csv'):
    df = pd.read_csv(path)
    df.drop(['Unnamed: 0','count','hate_speech','offensive_language','neither'],axis=1,inplace=True)
    return df


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

if choice == 'About Project':
    # jo krna ho krna
    pass
if choice == 'Detect HATE SPEECH':
    pass

if choice == 'View History':
    pass

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
    msg = st.sidebar.radio('select image',results)
    if msg and st.sidebar.button('delete msg'):
        if delete_message(msg.id):
            st.success("message deleted")
        else:
            st.error("message could not be deleted")