import sys
import os 
import pandas as pd
import numpy as np
import gensim
import pickle
from copy import deepcopy
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


model=gensim.models.Word2Vec.load("word2vec_t1.mdl")

df_trainq=pd.read_csv("trainq_t1.csv",sep='\t')
#df_testq=pd.read_csv("testq_t1.csv",sep='\t')
#df_testq['question1'] = df_testq['question1'].apply(lambda x: unicode(str(x),"utf-8"))
#df_testq['question2'] = df_testq['question2'].apply(lambda x: unicode(str(x),"utf-8"))
df_trainq['question1'] = df_trainq['question1'].apply(lambda x: unicode(str(x),"utf-8"))
df_trainq['question2'] = df_trainq['question2'].apply(lambda x: unicode(str(x),"utf-8"))

Y=np.array(df_trainq["is_duplicate"])
#np.save("Y",Y) 


# generate word weights
questions = list(df_trainq['question1']) + list(df_trainq['question2'])
tfidf = TfidfVectorizer(lowercase=False, )
tfidf.fit_transform(questions)
# dict key:word and value:tf-idf score
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf._tfidf.idf_))
del questions


#data transform for trainq question1
vecs1=[]
for stnc in tqdm(list(df_trainq['question1'])):
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
for stnc in tqdm(list(df_trainq['question2'])):
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
del df_trainq
X=np.concatenate((b1,list(vecs1)),axis=1)
np.save("X",X)
del X,b1,vecs1

with open('object.pickle','w') as f:
	pickle.dump([Y,  model, word2tfidf],f)


