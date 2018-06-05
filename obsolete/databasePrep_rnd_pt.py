import sys
import os 
import pandas as pd
import numpy as np
import gensim
from tqdm import tqdm
df = pd.read_csv("quora_duplicate_questions.tsv",delimiter='\t')
df=df.sample(frac=1)
df2 = pd.read_csv("unlabeledTrainData.tsv",header=0, delimiter='\t', quoting=3)
 
# encode questions to unicode
df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))
df2['review'] =df2['review'].apply(lambda x: unicode(str(x),"utf-8"))

#divide into train and test set
trainNum=int(df.shape[0]*0.88)
testNum=df.shape[0]-trainNum
df_trainq=df[:][:trainNum]
df_testq=df[:][trainNum:]










questions = list(df_trainq['question1']) + list(df_trainq['question2']) + list(df2['review'])     


# tokenize
c = 0
for question in tqdm(questions):
    questions[c] = list(gensim.utils.tokenize(question, deacc=True, lower=True))
    c += 1

# train model
model = gensim.models.Word2Vec(questions, size=100, workers=16, iter=10, negative=20)

# trim memory
model.init_sims(replace=True)

model.save('word2vec_t1.mdl')
model.wv.save_word2vec_format('word2vec_t1.bin',binary=True)
df_trainq.to_csv('trainq_t1.csv',sep='\t',encoding='utf-8')
df_testq.to_csv('testq_t1.csv',sep='\t',encoding='utf-8')
