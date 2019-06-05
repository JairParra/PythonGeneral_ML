""" DISCLAIMER: 
  All credits for this code to sentdex. See 
  https://www.youtube.com/watch?v=FLZvOKSCkxY&t=1s"""

### 11. Text Classification ### 

# Binary classification of sentiment : good or bad 
# Ex. spam filter 


import nltk
import random # to shuffle the dataset
from nltk.corpus import movie_reviews

# list of tuples, words are features
# (word, category )
documents  = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]

#documents = [()]
#for category in movie_reviews.categories():
#    for fileid in movie_reviews.fileids(category):
#        documents.append(list(movie_reviews.words(fileid),category))

random.shuffle(documents)

# Here we have a bunch of words and the last thing is the label
print(documents[1])

# We will take every word in every review and compile them, 
# and then we will take the most popular words, and then 
# we take which ones appear in positive and negative , 
# and then we search for those words. 

all_words = [] 
# add all the words for the test from all the reviews 
for w in movie_reviews.words(): 
    all_words.append(w.lower()) # case doesn't matter
    
# convert to an nltk frequency distribution 
all_words = nltk.FreqDist(all_words) # Frequency distance
print(all_words.most_common(15)) # most common words 
print(all_words["stupid" ]) # 253 times appears

### 12. Words as Features for Learning ### 

# We want to have a limit in the amount of words 

word_features = list(all_words.keys())[:3000] # up to the first 3000 words 

# within the document, it will find features
def find_features(document):
    words = set(document)
    features = {}
    # append to the dictionary if the word makes part 
    # of the top 4000 words
    for w in word_features:
        features[w] = (w in words) 

    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
# now we want to find features in the category
featuresets = [(find_features(rev), category) for (rev, category) in documents]


### 13. Naive Bayes ### 

training_set = featuresets[:1500] 
testing_set = featuresets[1500:]

# posterior = prior occurrences * likelihood / evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)


### 14. Save Classifier with Pickle ### 

# It is a way to saave python objects 

import pickle

classifier = nltk.NaiveBayesClassifier.train(training_set)

# save classifier
classifier_f  = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Naive Bayes accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

#save_classifier = open("naivebayes.pickle","wb") # wirte in bytes 
#pickle.dump(classifier, save_classifier)
#save_classifier.close()

# we could also add the whole document to pickle instead of running it each time 

# ********************************************************** #

### 15. Scikit-Learn incorporation ### 

import nltk
import random 
import pickle 
import numpy as np
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents  = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document) : 
    words = set(document) # turn into a unique set
    features = {}
    for w in word_features: 
        features[w] = (w in words)
        
    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


print("Original Naive Bayes accuracy: {}%".format(nltk.classify.accuracy(classifier, testing_set)*100))
classifier.show_most_informative_features(15)

# Multinomial bayes classifier 

# convert to nltk classifier with the wrapper 
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB Naive Bayes accuracy: {}%".format(nltk.classify.accuracy(MNB_classifier, testing_set)*100))

## Gaussian NB
#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set
#print("Gaussian Naive Bayes accuracy: {}%".format(nltk.classify.accuracy(GaussianNB_classifier, testing_set)*100))

# Bernoulli 
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("Bernoulli Naive Bayes accuracy: {}%".format(nltk.classify.accuracy(BernoulliNB_classifier, testing_set)*100))

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# Logistic Regression 
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression accuracy: {}%".format(nltk.classify.accuracy(LogisticRegression_classifier, testing_set)*100))

# SGDClassifier
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier accuracy: {}%".format(nltk.classify.accuracy(SGDClassifier_classifier, testing_set)*100))

# SVC
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC accuracy: {}%".format(nltk.classify.accuracy(SVC_classifier, testing_set)*100))

# LinearSVC 
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC accuracy: {}%".format(nltk.classify.accuracy(LinearSVC_classifier, testing_set)*100))

# NuSVC 
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("Bernoulli Naive Bayes accuracy: {}%".format(nltk.classify.accuracy(NuSVC_classifier, testing_set)*100))


print("Original Naive Bayes accuracy: {}%".format(nltk.classify.accuracy(classifier, testing_set)*100))
print("MNB Naive Bayes accuracy: {}%".format(nltk.classify.accuracy(MNB_classifier, testing_set)*100))
print("Bernoulli Naive Bayes accuracy: {}%".format(nltk.classify.accuracy(BernoulliNB_classifier, testing_set)*100))
print("LogisticRegression accuracy: {}%".format(nltk.classify.accuracy(LogisticRegression_classifier, testing_set)*100))
print("SGDClassifier accuracy: {}%".format(nltk.classify.accuracy(SGDClassifier_classifier, testing_set)*100))
print("SVC accuracy: {}%".format(nltk.classify.accuracy(SVC_classifier, testing_set)*100))
print("LinearSVC accuracy: {}%".format(nltk.classify.accuracy(LinearSVC_classifier, testing_set)*100))
print("Bernoulli Naive Bayes accuracy: {}%".format(nltk.classify.accuracy(NuSVC_classifier, testing_set)*100))

