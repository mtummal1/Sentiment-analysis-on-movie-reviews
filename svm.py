import pyprind
import pandas as pd
import os
import io
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
import threading
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.model_selection import cross_val_score
import time


df=pd.read_csv('./movie_review_data.csv')
snowball = SnowballStemmer('english')
stop = stopwords.words('english')
# vect=CountVectorizer(min_df=1)

def tokenizer_snowball(text):
    return [snowball.stem(word) for word in text.split()]

def tokenizer(text):
    return text.split()


train_data = df.loc[:10000, 'review'].values
train_labels= df.loc[:10000, 'sentiment'].values
test_data = df.loc[10000:, 'review'].values
test_labels = df.loc[10000:, 'sentiment'].values
# X=np.concatenate(X_train,y_train)

# train_data , train_labels, test_data, test_labels = train_test_split(df.loc[:1000, 'review'].values,df.loc[:1000, 'sentiment'].values, test_size=0.4, random_state=0)

vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

# lr_tfidf = Pipeline([('vect', tfidf),
                 # ('clf', svm.SVC(kernel='linear'))])


classifier_rbf = svm.SVC()
t0 = time.time()
classifier_rbf.fit(train_vectors, train_labels)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(test_vectors)
t2 = time.time()
time_rbf_train = t1-t0
time_rbf_predict = t2-t1
# print(X_t)
# score=cross_val_score(lr_tfidf, X_t.toarray(),y_train, cv=1)
# lr_tfidf.best_score_
# clf = lr_tfidf.best_estimator_







print(score)
