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
import cPickle as pkl


# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words, skip_top=20)
# truncate and pad input sequences

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 100

def create_model(neurons=8):
	# create model
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(neurons))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=1)
#print(model.summary())
# define the grid search parameters

neurons = [ 16, 32, 64, 128]
batch_size = [64]
epochs = [4]

param_grid = dict(neurons=neurons, epochs=epochs, batch_size=batch_size)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

best_model=create_model(grid_result.best_params_['neurons'])
best_model.fit(X_train, y_train, epochs=grid_result.best_params_['epochs'], batch_size=grid_result.best_params_['batch_size'])
# Final evaluation of the model

scores = best_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

f = open('CVresultsBasic.pkl', 'wb')
pkl.dump(grid_result.best_params_,f,pkl.HIGHEST_PROTOCOL)
pkl.dump(grid_result.cv_results_,f,pkl.HIGHEST_PROTOCOL)
f.close()

best_model.save('modelBasic.h5')


#model = load_model('data/model.h5')
#scores = model.evaluate(X_test, y_test, verbose=1)
#print("Accuracy: %.2f%%" % (scores[1]*100))

#f = open('data/CVresults.pkl', 'rb')
#for i in range(2):
#    print pkl.load(f)

#f.close()