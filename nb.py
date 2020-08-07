import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import scipy.sparse
import numpy as np
from matplotlib import pyplot as plt

"""
moviedir = r'data_full'
movie_train = load_files(moviedir, shuffle=True)

movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
movie_counts = movie_vec.fit_transform(movie_train.data)

tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(movie_counts)

docs_train, docs_test, y_train, y_test = train_test_split(
    movie_tfidf, movie_train.target, test_size = 0, random_state = 12)



scipy.sparse.save_npz('movie_train_tfidf_data.npz', docs_train)
np.save('movie_train_tfidf_data_label.npy',y_train)
"""


docs_train_reload = scipy.sparse.load_npz('movie_train_tfidf_data.npz')
y_train_reload = np.load('movie_train_tfidf_data_label.npy')


frac_data = [0.1, 0.3, 0.6, 0.9, 1]
accuracy = []

doc_test = docs_train_reload[20000:25000]
y_test = y_train_reload[20000:25000]

for i in frac_data:

	doc_train = docs_train_reload[0:int(i*20000)]
	y_train = y_train_reload[0:int(i*20000)]

	clf = MultinomialNB().fit(doc_train, y_train)

	y_pred = clf.predict(doc_test)
	accuracy.append(sklearn.metrics.accuracy_score(y_test, y_pred))


plt.title('Plot of Accuracy vs Fraction of training data')
plt.ylabel('Accuracy')
plt.xlabel('Fraction of Training data')
plt.scatter(frac_data,accuracy)
plt.plot(frac_data,accuracy)
plt.show()
plt.savefig("figure")