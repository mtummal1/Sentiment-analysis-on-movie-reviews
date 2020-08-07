import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import scipy.sparse
import numpy as np

moviedir = r'data_full_25k'
movie_train = load_files(moviedir, shuffle=True)

movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
movie_counts = movie_vec.fit_transform(movie_train.data)

tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(movie_counts)

docs_train, docs_test, y_train, y_test = train_test_split(
    movie_tfidf, movie_train.target, test_size = 0, random_state = 12)



scipy.sparse.save_npz('movie_train_tfidf_data.npz', docs_train)
np.save('movie_train_tfidf_data_label.npy',y_train)






