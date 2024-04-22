from flask import Flask,redirect,url_for,render_template,request
import os
import pandas as pd
import numpy as np
import pickle
import re     
from sklearn.feature_extraction.text import CountVectorizer                             
import nltk  
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')

from nltk.corpus import stopwords           
from nltk.stem.porter import PorterStemmer 

ps=PorterStemmer()

all_stopwords=stopwords.words('english')    
all_stopwords.remove('not')

app=Flask(__name__)
pic=os.path.join('static','images')
app.config['UPLOAD_FOLDER']=pic

@app.route('/')
def home():
    pic2=os.path.join(app.config['UPLOAD_FOLDER'],'submit.jpg')
    return render_template('homepage.html',user_image2=pic2)

@app.route('/review')
def review():
    return render_template('reviewpage.html')

@app.route('/submit',methods=['POST','GET'])
def submit():
    r=''
    if request.method=='POST':
        r=request.form['reviewtxt']
    df2=open(r'newdata.txt','a+')
    df2.write(r+'\n')
    df2.close()
    return redirect(url_for('review'))

@app.route('/view')
def view():
    df2=open(r'newdata.txt','r')
    nr=df2.readlines()
    df2.close()
    saved_model = pickle.load(open("Classifier_Sentiment_Model.pkl",'rb'))
    cleaned_data=clean(nr)
    predictions=saved_model.predict(cleaned_data)
    p=0
    n=0
    for c in predictions:
      if(c=='Positive'):
        p=p+1
      else:
        n=n+1
    sumx=p+n
    exp={}
    for i in range(len(predictions)):
      exp[nr[i]]=predictions[i]
    return render_template('viewpage.html',pre=predictions,px=p,nx=n,vw=nr,ex=exp,sum1=sumx)
def clean(raw_data):
  reviews=[]
  for i in raw_data:
    review = re.sub('[^a-zA-Z]', ' ',i)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    reviews.append(str(review))
  vectors = vectorizer(reviews)
  return vectors
def vectorizer(cleaned_data):
  cvFile='bow_sentiment model.pkl'  
  cv = pickle.load(open(cvFile, "rb"))
  vec = cv.transform(cleaned_data).toarray()
  return vec
if __name__ == "__main__":
    app.run(debug=True)