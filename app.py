from flask import Flask, render_template,request
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

data=pd.read_csv('spam.csv')
data['spam']=data['Category'].apply(lambda x : 1 if x=='ham' else 0)
x_train,x_test,y_train,y_test=train_test_split(data['Message'],data['Category'],test_size=0.20,random_state=2)

vector=CountVectorizer()
model=MultinomialNB()
x_count=vector.fit_transform(x_train)
model.fit(x_count,y_train)


app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_email():
    email=request.form.get('email')
    variable=vector.transform([email])
    result=model.predict(variable)[0]
    if result=="ham":
        result= "THIS IS A HAM MAIL. SAFE TO OPEN"
    else:
        result= "SPAM MAIL ALERT !!!!"
    return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(debug=True)