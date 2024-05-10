#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify, render_template
import pickle as pkl
import scipy.sparse
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

with open('Decision_Tree.pkl', 'rb') as file:
    model = pkl.load(file)
    
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pkl.load(file)

def predict_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    tokens = nltk.word_tokenize(text)
    
    # removing stopwords
    y = []
    for token in tokens:
        if token not in stopwords.words('english'):
            y.append(token)
    text = " ".join(y[1:])  
    
    text_features = vectorizer.transform([text]) 
    
    print(model)
    
    prediction = model.predict(text_features)
    propensity = model.predict_proba(text_features)[0][1]  # Probability of the spam class
    return {"prediction": str(prediction[0]), "propensity": float(propensity)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score', methods = ['POST'])
def score():
    data = request.json
    text = data.get('text', '')
    result = predict_text(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




