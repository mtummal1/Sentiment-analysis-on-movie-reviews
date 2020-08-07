# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import cPickle as pkl


# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 10000
skip_top = 20
vocab_size = top_words - skip_top
index_from=3   # word index offset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words, skip_top=skip_top, index_from=index_from)
# truncate and pad input sequences

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32

word_to_id = imdb.get_word_index()
word_to_id = {k:(v+index_from) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = numpy.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print len(embeddings_index)

print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = numpy.zeros((top_words+index_from, 100))
for word, i in word_to_id.items():
  word = word.replace("'","")
    
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None and i < top_words+index_from:
    embedding_matrix[i] = embedding_vector

    
print numpy.where(~embedding_matrix.any(axis=1))[0]


def create_model(neurons=8):
	# create model
    model = Sequential()
    e = Embedding(top_words+index_from, 100, weights=[embedding_matrix], input_length=max_review_length, trainable=True)
    model.add(e)
    #model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(neurons))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=1)
#print(model.summary())
# define the grid search parameters

#### Actual########
#neurons = [8, 16, 64, 128]
#batch_size = [64, 128, 256, 512]
#epochs = [4, 6, 8]

# test
neurons = [16, 32, 64, 128]
batch_size = [64]
epochs = [4]

param_grid = dict(neurons=neurons, epochs=epochs, batch_size=batch_size)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

best_model=create_model(grid_result.best_params_['neurons'])
best_model.fit(X_train, y_train, epochs=grid_result.best_params_['epochs'], batch_size=grid_result.best_params_['batch_size'])

scores = best_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



best_model=create_model(grid_result.best_params_['neurons'])
best_model.fit(X_train, y_train, epochs=grid_result.best_params_['epochs'], batch_size=grid_result.best_params_['batch_size'])
# Final evaluation of the model

scores = best_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


f = open('CVresults.pkl', 'wb')
pkl.dump(grid_result.best_params_,f,pkl.HIGHEST_PROTOCOL)
pkl.dump(grid_result.cv_results_,f,pkl.HIGHEST_PROTOCOL)
f.close()


best_model.save('model.h5')