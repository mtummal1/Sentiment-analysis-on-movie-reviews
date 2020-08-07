1. Download the data set from this link
https://ai.stanford.edu/~amaas/data/sentiment/

2. extract the folder and go to aclimdb/train
copy the neg and pos folder and paste it in an another folder 'data_full_25k'

3. now run the data_process.py program
python data_process.py

note - the data_full_25k folder should be present in the same folder where you run the python data_process.py

4. you will see two files created
- movie_train_tfidf_data.npz
- movie_train_tfidf_data_label.npy

To run the following code type
python filename.py
the two files generated should be present in the same folder where you run your python program 
if any module is not installed then it will through an error,just install that module
