import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
import re
import os
import pickle
print("1")
#cwd = os.getcwd()  # Get the current working directory (cwd)
#files = os.listdir(cwd)  # Get all the files in that directory
#print("Files in %r: %s" % (cwd, files))

files_pos = os.listdir('Resources/train/pos')
files_pos = [open('Resources/train/pos/'+f, 'r', encoding="utf8").read() for f in files_pos]
files_neg = os.listdir('Resources/train/neg')
files_neg = [open('Resources/train/neg/'+f, 'r', encoding="utf8").read() for f in files_neg]

all_words = []
documents = []
print("2")
from nltk.corpus import stopwords


stop_words = list(set(stopwords.words('english')))

allowed_word_types = ["J"]
print("3")
for p in files_pos:
    documents.append((p, "pos"))
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    tokenized = word_tokenize(cleaned)
    stopped = [w for w in tokenized if not w in stop_words]
    pos = nltk.pos_tag(stopped)
    
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
print("4")            
for p in files_neg:
    documents.append((p, "neg"))
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    tokenized = word_tokenize(cleaned)
    stopped = [w for w in tokenized if not w in stop_words]
    neg = nltk.pos_tag(stopped)
    
    for w in neg:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
print("5")            
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:1000]
save_wordfeats = open("wordfeats1.pickle","wb")
pickle.dump(word_features, save_wordfeats)
save_wordfeats.close()
print("6")
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print("7")
featuresets = [(find_features(rev), category) for (rev, category) in documents]
#save_featset = open("featset1.pickle","wb")
#pickle.dump(featuresets, save_featset)
#save_featset.close()
print("7.1")
random.shuffle(featuresets)
print("7.2") 
training_set = featuresets[:100]
print("7.3")
testing_set = featuresets[100:]
#save_testingset = open("testingset1.pickle","wb")
#pickle.dump(testing_set, save_testingset)
#save_testingset.close()
print("7.4")
print("8")

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
print("9")
classifier.show_most_informative_features(15)
print("10")

#save_classifier = open("naivebayes.pickle","wb")
#pickle.dump(classifier, save_classifier)
#save_classifier.close()


#Directions
#First run the display.py in order to create and save all the pickle files.
#Then run the sentiment_mod.py to load all the pickles and run the sentiment algorithm.
#Then run the Test1.py with the input text in order to run it against the trained AI and sentiment.