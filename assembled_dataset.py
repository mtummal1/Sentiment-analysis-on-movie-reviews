import pandas as pd
import os
import io
import numpy as np

basepath='./aclImdb'
labels={'pos':1, 'neg':0}
pbar=pyprind.ProgBar(500000)
df=pd.DataFrame()
for s in ('test','train'):
	for l in ('pos','neg'):
		path=os.path.join(basepath,s,l)
		for file in os.listdir(path):
			with open(os.path.join(path,file),'r',encoding='utf-8')as infile:
				txt=infile.read()
			df=df.append([[txt,labels[l]]],ignore_index=True)
			pbar.update()
	print('\n')
df.columns=['review','sentiment']
x=df.loc[:,:].values
#print(x)
np.random.seed(0)
df=df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_review_data.csv',index=False,encoding='utf-8')
