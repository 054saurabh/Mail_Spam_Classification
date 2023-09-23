from flask import Flask,render_template,request,flash
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
vect=pickle.load(open('vectorization.pkl','rb'))


def text_transform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():# is alpha numeric
            y.append(i)
            
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result',methods=['post'])
def result():
    # take input
    message=request.form['message']
    # preprocessing
    trns_mess=text_transform(message)
    # vectorize
    vectorise=vect.transform([trns_mess])
    # predict
    predict=model.predict(vectorise)[0]
    if predict==1:
        output="Spam"
    elif predict==0:
        output="Not Spam"

    # display
    return render_template('index.html',n=output)


if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080)