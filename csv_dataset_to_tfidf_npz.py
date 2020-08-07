import pyprind
import pandas as pd
import os
import io
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# my_data = np.genfromtxt('/home/shubham/Desktop/sml/nlp/movie_review_data.csv')
# print(my_data.shape)
df=pd.read_csv('./movie_review_data.csv')
snowball = SnowballStemmer('english')
stop = stopwords.words('english')
vect=CountVectorizer(min_df=1)
transformer = TfidfTransformer(smooth_idf=False)
vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf =True, stop_words = 'english')
# print(type(df.loc[0,'review']))
# text=df.loc[0,'review']
# print(text,'\n**************************************************************************************************************')
# # print(df.head(0),df.shape)
# # data_np=df.as_matrix()
# # data_np.astype('object')
# # print(data_np[:,1])
# # print(df)
# # text=data_np[:,0]
# text=re.sub('<[^>]*>','',text)
# emot=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
# text = re.sub('[\W]+', ' ', text.lower()) +\
#         ' '.join(emot).replace('-', '')


def clean_up(a_review):	# removes html and adds punctuations as well as emoticons at the back of the sentense
	
	a_review=re.sub('<[^>]*>','',a_review)
	# print(len(a_review))
	# print(len(a_review))
	emot=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',a_review)
	a_review = re.sub('[\W]+', ' ', a_review.lower()) +\
        ' '.join(emot).replace('-', '')
	return ' '.join([sw for sw in [snowball.stem(word) for word in a_review.split()] if sw not in stop])
	# return a_review.split()

def main():
	review_string=[]
	y_test = df.loc[:, 'sentiment'].values
	y_test_t=y_test.reshape(-1,1)
	print(y_test,'\n',y_test.reshape(-1,1).shape)
	for i in np.arange(50000):
		# print(df.loc[0,'review'],'\n***************************************************************\n')
		review_string.append(clean_up(df.loc[i,'review']))
	# print(review_string)
	# print(review_string[0])
	# print(review_string,'\n')
	np.set_printoptions(precision=3)
	# bag=vect.fit_transform(review_string)
	# print(bag.shape,'\n')
	# print(bag.toarray().shape)
	# tfidf = transformer.fit_transform(vect.fit_transform(review_string).toarray())
	tfidf = vectorizer.fit_transform(review_string) #does both bag of words and tfidf conversion of cleaned up data

	# print(tfidf.toarray())
	# a=np.concatenate((tfidf.toarray(),y_test_t[0:1000]),axis=1)
	# print('a shape final',a.shape,'\n')
	np.savez_compressed("tfidf_movie_review.npz", np.array(tfidf), delimiter=",")# converts rfidf data to .npz file so that we can use it to provide quick access to the data to train
	# df['review']=pd.Series(bag)
	# print(df.loc[0,'review'],'\n********************************')
	# print(df.loc[1,'review'],'\n********************************')
main()
