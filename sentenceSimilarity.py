import sys
import os 
import pandas as pd
import numpy as np
import gensim
#from tqdm import tqdm
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

model=[]
word2tfidf=[]
forest=[]

def loadWord2VecModel():
 	global model
 	model=gensim.models.Word2Vec.load("word2vec_t1.mdl")

def loadWord2tfidf():
	global word2tfidf
	with open('word2tfidf.pickle') as f:
		word2tfidf=pickle.load(f)
	word2tfidf=word2tfidf[0] #due to issue when saving
	
def loadRandomForestClassifier():
	global forest
	with open('randomForestModel25.pickle') as f:
		forest=pickle.load(f)

def tempTrainWord2tfidf():
	df_trainq,Y=tempLoadQuestionDuplicatePair()
	word2tfidf=generateWordWeights(list(df_trainq['question1']) + list(df_trainq['question2']))
	with open('word2tfidf.pickle','w') as f:
		pickle.dump([word2tfidf],f)

def tempPrepDataforDecisionTreeTraining():
#if(True):
	#This function will cause memory issue on this computer
	#needs to have word2tfidf and word2vec model loaded
	df_trainq,_=tempLoadQuestionDuplicatePair()
	vec1=dataTransform(df_trainq["question1"],word2tfidf)
	vec2=dataTransform(df_trainq["question2"],word2tfidf)
	vec1=np.float32(vec1)
	vec2=np.float32(vec2)
	X=np.float32(np.concatenate((vec1,vec2),axis=1))
	np.save("decisionTreeTrainingData",X)


def tempTrainWord2Vec():
	df = pd.read_csv("quora_duplicate_questions.tsv",delimiter='\t')
	df=df.sample(frac=1)
	df2 = pd.read_csv("unlabeledTrainData.tsv",header=0, delimiter='\t', quoting=3)
 
	# encode questions to unicode
	df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
	df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))
	df2['review'] =df2['review'].apply(lambda x: unicode(str(x),"utf-8"))
	df_trainq=df[:][:]
	questions = list(df_trainq['question1']) + list(df_trainq['question2']) + list(df2['review'])     

	# tokenize
	c = 0
	for question in tqdm(questions):
    		questions[c] = list(gensim.utils.tokenize(question, deacc=True, lower=True))
    		c += 1

	# train model
	model = gensim.models.Word2Vec(questions, size=300, workers=16, iter=10, negative=20)

	# trim memory
	model.init_sims(replace=True)
	model.save('word2vec_t1.mdl')
	model.wv.save_word2vec_format('word2vec_t1.bin',binary=True)
	df_trainq.to_csv('trainq_t1.csv',sep='\t',encoding='utf-8')
	
	
def tempLoadQuestionDuplicatePair():
	df_trainq=pd.read_csv("trainq_t1.csv",sep='\t')
	df_trainq['question1'] = df_trainq['question1'].apply(lambda x: unicode(str(x),"utf-8"))
	df_trainq['question2'] = df_trainq['question2'].apply(lambda x: unicode(str(x),"utf-8"))
	Y=np.array(df_trainq["is_duplicate"])
	return df_trainq,Y

def tempRandomForestTrain():
	X=np.load("decisionTreeTrainingData.npy")
	_,Y=tempLoadQuestionDuplicatePair()
	forest=RandomForestClassifier(n_estimators=5)
	forest=forest.fit(X,Y)
	with open('randomForestModel.pickle','w') as f:
		pickle.dump(forest,f)

def generateWordWeights(questions):
	# generate word weights
	tfidf = TfidfVectorizer(lowercase=True, )
	tfidf.fit_transform(questions)
	
	# dict key:word and value:tf-idf score
	word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf._tfidf.idf_))
	return word2tfidf

def dataTransform(dataStrings,word2tfidf):
	global model
	vecs1=[]
	for stnc in tqdm(list(dataStrings)):
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
	return vecs1

def featureExtract(stnc):
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
	return mean_vec

def testing():
	Q,Y=tempLoadQuestionDuplicatePair()
	nsum=0
	for f in xrange(len(Q)):
		print f, nsum/(f+1e-10)
		res=classify(Q["question1"][f],Q["question2"][f])
		if(res==Y[f]):
			nsum+=1.
	print nsum/len(Q)

def classify(str1,str2):
	#random forest model needs to be loaded first
	global forest
	a=featureExtract(str1)
	b=featureExtract(str2)
	c=np.concatenate((a.reshape(1,-1),b.reshape(1,-1)),axis=1)
	return forest.predict(c)
	
def classifyProb(str1,str2):
	#random forest model needs to be loaded first
	global forest
	a=featureExtract(str1)
	b=featureExtract(str2)
	c=np.concatenate((a.reshape(1,-1),b.reshape(1,-1)),axis=1)
	return forest.predict_proba(c)

##For training
tempTrainWord2Vec() #train word embedding
tempTrainWord2tfidf() #obtain idf
tempPrepDataforDecisionTreeTraining() #data preparation for classifier
tempRandomForestTrain() #random forest training

#For using
loadWord2VecModel() #load word embedding model
loadWord2tfidf()  #load idf dictionary
loadRandomForestClassifier() #load trained random forest classifier
print classifyProb("question1","question2")


