The program is to train a classifier that determines whether two questions are similar or not.

Dataset are obtained from Quora dataset (https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) and unlabeled IMDB review dataset (https://www.kaggle.com/c/word2vec-nlp-tutorial/data) 

sentenceSimilarity.py contains code for training (word embedding, idf and random forest classifier) and using the model.

SiameseNet.py trains a siamese network classifier in PyTorch (it needs data from decisionTreeTrainingData.npy, which can be generated from tempPrepDataforDecisionTreeTraining()function in sentenceSimilarity.py).

biRnnCpu_Train.py trains bi-drectional LSTM simases network (requires word embedding model from sentenceSimilarity.py)

