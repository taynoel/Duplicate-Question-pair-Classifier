import sys
import os 
import pandas as pd
import numpy as np
import gensim
import pickle
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

#load and train random forest classifier
X=np.load("X.npy")
with open('object.pickle') as f:
	Y,model,word2tfidf=pickle.load(f)

forest=RandomForestClassifier(n_estimators=10)
forest=forest.fit(X,Y)
del X,Y

with open('object_forest.pickle','w') as f:
	pickle.dump(forest,f)



	


#start testing
#load testing data
df_testq=pd.read_csv("testq_t1.csv",sep='\t')
df_testq['question1'] = df_testq['question1'].apply(lambda x: unicode(str(x),"utf-8"))
df_testq['question2'] = df_testq['question2'].apply(lambda x: unicode(str(x),"utf-8"))


#data transform for trainq question1
vecs1=[]
for stnc in tqdm(list(df_testq['question1'])):
	stnc_Token=list(gensim.utils.tokenize(stnc, deacc=True, lower=True))
	mean_vec=np.zeros(300)
	mean_idf=0
	
	for word in stnc_Token:
		try:
			wordVec=model[str(word)]
		except:
			wordVec=np.zeros(300)
		
		try:
			idf=word2tfidf[str(word)]
		except:
			idf=0;
		mean_vec+=wordVec*idf
		mean_idf+=idf
	if mean_idf !=0: 
		mean_vec/=mean_idf
	else:
		mean_vec/=0.000000001
	vecs1.append(mean_vec)
#df_trainq['q1_feats']=list(vecs1)
b1=list(vecs1)


#data transform for trainq question2
vecs1=[]
for stnc in tqdm(list(df_testq['question2'])):
	stnc_Token=list(gensim.utils.tokenize(stnc, deacc=True, lower=True))
	mean_vec=np.zeros(300)
	mean_idf=0
	
	for word in stnc_Token:
		try:
			wordVec=model[str(word)]
		except:
			wordVec=np.zeros(300)
		
		try:
			idf=word2tfidf[str(word)]
		except:
			idf=0;
		mean_vec+=wordVec*idf
		mean_idf+=idf
	if mean_idf !=0: 
		mean_vec/=mean_idf
	else:
		mean_vec/=0.000000001
	vecs1.append(mean_vec)

T=np.concatenate((b1,list(vecs1)),axis=1)


#Performance test (basic accuracy)
NumSamples=np.shape(T)[0]
GroundTruth=df_testq["is_duplicate"][:]

c=0
Correct=0
for w in T:
	res=forest.predict(w)
	if(res==GroundTruth[c]):
		Correct+=1
	c+=1


