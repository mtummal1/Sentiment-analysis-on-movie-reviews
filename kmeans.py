import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import scipy.sparse
from matplotlib import pyplot as plt



docs_train_reload = scipy.sparse.load_npz('movie_train_tfidf_data.npz')
y_train_reload = np.load('movie_train_tfidf_data_label.npy')

dat_size = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
error = []
for i in dat_size:
	docs_train_reload=docs_train_reload[0:int(i*25000)]
	y_train_reload=y_train_reload[0:int(i*25000)]
	model = KMeans(n_clusters=2)
	model = model.fit( docs_train_reload )
	print('error is :')
	temp = np.sum(model.labels_ == y_train_reload)/int(i*25000)*100
	print(temp)
	error.append(temp)



plt.title('Plot of error vs size of training data')
plt.ylabel('error')
plt.xlabel('size of training data')
plt.scatter(dat_size,error)
plt.plot(dat_size,error)
plt.show()
plt.savefig("figure")




