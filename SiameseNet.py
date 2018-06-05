import numpy as np
import torch
import torch.utils.data
import pickle
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(300, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 100)
        self.fc7 = nn.Linear(100, 100)
        self.fc8 = nn.Linear(100, 20)
        self.fc9 = nn.Linear(20, 10)

    def forward(self, x1,x2):
        xa = F.relu(self.fc1(x1))
        xa = F.relu(self.fc5(xa))
        xa = F.relu(self.fc6(xa))
        xa = F.relu(self.fc7(xa))
        xa = F.relu(self.fc8(xa))
        y1 = self.fc9(xa)
        xb = F.relu(self.fc1(x2))
        xb = F.relu(self.fc5(xb))
        xb = F.relu(self.fc6(xb))
        xb = F.relu(self.fc7(xb))
        xb = F.relu(self.fc8(xb))
        y2 = self.fc9(xb)
        return y1,y2

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

if False:
    X=np.load("decisionTreeTrainingData.npy")
    df_trainq=pd.read_csv("trainq_t1.csv",sep='\t')
    Y=np.array(df_trainq["is_duplicate"])
    
    #Temporary
    Xt=X[:350000]
    del X
    Yt=Y[:350000]
    del Y
    
    Xtrain=[[f,int(Yt[f1])] for f1,f in enumerate(Xt)]
    trainloader = torch.utils.data.DataLoader(Xtrain, batch_size=512,shuffle=True, num_workers=2)
    del Xtrain
    #with open('TrainLoader.pickle', 'wb') as fn:
       #pickle.dump(trainloader,fn)

net = Net()
#criterion = nn.CrossEntropyLoss()
criterion = ContrastiveLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001) #0.001

def findAccuracy(o1,o2,labels):
    diff=o1-o2
    dist_sq = torch.sum(torch.pow(diff, 2), 1)
    correct=0.0
    for f,v in enumerate(dist_sq):
        if v.item()>0.1: ress=0.0
        else: ress=1.0
        if ress==labels.data[f].item(): correct+=1.
    return correct/float(labels.size()[0])
            
    
    
#For training    
if True:
   #Load model weight
   net.load_state_dict(torch.load("model3.dict"))

   trainLoss=[]
   for epoch in range(2000):
       sumLoss=0.0
       Count=0
       Accuracy=0.0
       for i, data in enumerate(trainloader, 0):
           inputsLarge, labels = data
           input1=inputsLarge[:,:300]
           input2=inputsLarge[:,300:]
           input1,input2,labels = Variable(input1),Variable(input2), Variable(labels.type(torch.FloatTensor))
           optimizer.zero_grad()
           output1,output2 = net(input1,input2)
           loss = criterion(output1,output2, labels)
           loss.backward()
           #print net.fc1.weight.data[0][0],net.fc1.weight.grad.data[0][0]
           optimizer.step()
           sumLoss+=loss.item()
           Accuracy+=findAccuracy(output1,output2,labels)
           Count+=1

       print "Loss", sumLoss/Count,"Accuracy",Accuracy/Count,"at",epoch
       trainLoss.append(loss.item())
   #torch.save(net.state_dict(),"model3.dict")
