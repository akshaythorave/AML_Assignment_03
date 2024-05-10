#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords


# In[24]:


def score(text, model, threshold):
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

    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    vectorized_text = vectorizer.transform([text])

    propensity = model.predict_proba(vectorized_text)
    if propensity[0][1] >= threshold: 
        prediction = 1
        propen = propensity[0][1]
    else: 
        prediction = 0
        propen = propensity[0][0]
    return prediction, propen


# In[ ]:




