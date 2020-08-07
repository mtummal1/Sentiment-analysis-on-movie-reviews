1]$ wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz & $ tar xvzf aclImdb_v1.tar.gz use those commands to download and extract dataset
2]Run assembled_dataset.py to generate consolidated dataset file movie_review_data.csv , this may take some time.
3]Run csv_dataset_to_tfidf_npz.py to convert movie_review_data.csv into tfidf_movie_review_dataset.npz, you can use this tfidf dataset to run any of your model.
4]sv.py contains code for 5 fold cross validation of the svm model onto the movie_review_data.csv so run this after step 1.
