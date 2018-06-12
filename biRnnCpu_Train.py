import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import gensim

LoadPretrainedFlag=1

class CdescpLstm(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(CdescpLstm, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim*2, 50)
        self.fc2 = nn.Linear(50, 4)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        sentenceFlip=torch.from_numpy(np.flip(sentence.numpy(),0).copy())
        self.hidden = self.init_hidden()
        lstm_out1, self.hidden = self.lstm(sentence, self.hidden)
        self.hidden = self.init_hidden()
        lstm_out2, self.hidden = self.lstm(sentenceFlip, self.hidden)
        lstm_out=torch.cat((lstm_out1[len(lstm_out1)-1],lstm_out2[len(lstm_out2)-1]),1)
        fco = self.fc1(lstm_out[len(lstm_out)-1])
        fco = F.relu(fco)
        fco = self.fc2(fco)
        return fco

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq+1e-16)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

word2VecModel=[]
def loadWord2VecModel():
 	global word2VecModel
 	word2VecModel=gensim.models.Word2Vec.load("word2vec_t1.mdl")

def tempLoadQuestionDuplicatePair():
	df_trainq=pd.read_csv("trainq_t1.csv",sep='\t')
	df_trainq['question1'] = df_trainq['question1'].apply(lambda x: unicode(str(x),"utf-8"))
	df_trainq['question2'] = df_trainq['question2'].apply(lambda x: unicode(str(x),"utf-8"))
	#Y=np.array(df_trainq["is_duplicate"])
	return df_trainq

def tempTrainDataloaderPreparation():
    #df_dataList[TrainDataIndex][0:sentence,1:label]
    #df_dataList[TrainDataIndex][0][0:first sentence,1:second sentence]
    trainIndex=350000
    df_trainq=tempLoadQuestionDuplicatePair()
    df_1stStnc=df_trainq["question1"][:trainIndex]
    df_2ndStnc=df_trainq["question2"][:trainIndex]
    df_dataList=[[[list(gensim.utils.tokenize(stnc1, deacc=True, lower=True)),
         list(gensim.utils.tokenize(df_2ndStnc[stncN], deacc=True, lower=True))],
         df_trainq["is_duplicate"][stncN]] 
         for stncN,stnc1 in enumerate(df_1stStnc) ]
    trainIndexData=[[f,df_trainq["is_duplicate"][f]] for f in xrange(trainIndex)]
    trainloader = torch.utils.data.DataLoader(trainIndexData, batch_size=256,shuffle=True, num_workers=2)
    return trainloader,df_dataList

def sentence2VecList(stnc):
    return [word2VecModel.wv[s] for s in stnc if s in word2VecModel.wv]

loadWord2VecModel()
net = CdescpLstm(300,50)
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001) #0.001
if LoadPretrainedFlag:
    net.load_state_dict(torch.load("modelLSTM.dict"))

#Training
if True:
    trainloader,df_dataList=tempTrainDataloaderPreparation()
    for epoch in range(100):
        lossNumforPrint=0.0
        for f1, data in enumerate(trainloader, 0):
            print "at Batch",f1
            dataIndices,labels=data
            lossListInBatch=[]
            for f2,dataIndex in enumerate(dataIndices):
                label=df_dataList[dataIndex.item()][1]
                stnc1=df_dataList[dataIndex.item()][0][0]
                stnc2=df_dataList[dataIndex.item()][0][1]
                stnc1T=torch.tensor(sentence2VecList(stnc1)).unsqueeze(1)
                stnc2T=torch.tensor(sentence2VecList(stnc2)).unsqueeze(1)
                output1=net(stnc1T).unsqueeze(0)
                output2=net(stnc2T).unsqueeze(0)
                lossListInBatch.append(criterion(output1,output2,float(label)))
            net.zero_grad()
            lossTotal=torch.tensor(0.0)
            for f in xrange(len(lossListInBatch)-1):
                lossTotal+=lossListInBatch[f]
            lossNumforPrint+=lossTotal.item()
            lossTotal.backward()
            optimizer.step()
            print lossTotal
            torch.save(net.state_dict(),"modelLSTM.dict")
        print lossNumforPrint
            
    