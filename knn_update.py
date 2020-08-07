import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import scipy.sparse
import numpy as np
from matplotlib import pyplot as plt

docs_train_reload = scipy.sparse.load_npz('movie_train_tfidf_data.npz')
y_train_reload = np.load('movie_train_tfidf_data_label.npy')

k_value = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
accuracy = []

doc_train = docs_train_reload[0:20000]
y_train = y_train_reload[0:20000]

doc_test = docs_train_reload[20000:25000]
y_test = y_train_reload[20000:25000]

for i in k_value:

	neigh = KNeighborsRegressor(n_neighbors=i)
	neigh.fit(doc_train, y_train) 
	target = neigh.predict(doc_test)
	accuracy.append(100-(np.sum(target == y_test)/5000*100))




plt.title('Plot of Accuracy vs K value')
plt.ylabel('Accuracy')
plt.xlabel('K value')
plt.scatter(k_value,accuracy)
plt.plot(k_value,accuracy)
plt.show()
plt.savefig("figure")
