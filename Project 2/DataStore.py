import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import re
import os
import pickle

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

testingset_f = open("testingset1.pickle","rb")
testingset = pickle.load(testingset_f)
testingset_f.close()

featset_f = open("featset1.pickle","rb")
featset = pickle.load(featset_f)
featset_f.close()

wordfeats_f = open("wordfeats1.pickle", "rb")
word_features = pickle.load(wordfeats_f)
wordfeats_f.close()

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

testing_set = featset[100:]
training_set = featset[:100]


voted_classifier = VoteClassifier(classifier)
print ("Voted Classifier % ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
    feats = find_features(text)
    
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)