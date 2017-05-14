# Duplicate-Question-pair-Classifier

databasePrep_rnd.py:
Dataset are obtained from Quora dataset (https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) and unlabeled IMDB review dataset (https://www.kaggle.com/c/word2vec-nlp-tutorial/data)
Prepares the training and testing dataset 
Using training and additional unlabeled dataset, word2vec is generated too (Quora training dataset + unlabeled IMDB review dataset)


featureTransform.py:
Transforms the data in a way that fits for training (word to vector conversion and word significance weighting)

RFtest.py:
Random forest classifier training and testing

For those ending with "pt", the feature vector of a sentence is divided into 3 partitions. Word vecs (reduced to 100 elements) will be summed to their respective partition in a sentence. No noticable improvement is observed  
