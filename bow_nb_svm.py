import pandas as pd
import numpy as np
#import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from tokenizer import tokenize
#import nltk
#nltk.download('punkt')

#training data
data_train = pd.read_csv('train.tsv',delimiter='\t')

# replace label with numbers
# categorize = {'label': {'TRUE':1, 'mostly-true':2, 'half-true':3, 'barely-true':4, 'pants-fire':5, 'FALSE':6}}
# data_train.replace(categorize,inplace=True)

X_train = pd.DataFrame(data=data_train, columns = ['statement'])
y_train = pd.DataFrame(data=data_train, columns = ['label'])

X_train = X_train.statement.tolist()
y_train = np.array(y_train.label)

#Testing data
data_test = pd.read_csv('test.tsv',delimiter='\t')

# replace label with numbers
# categorize = {'label': {'TRUE':1, 'mostly-true':2, 'half-true':3, 'barely-true':4, 'pants-fire':5, 'FALSE':6}}
# data_test.replace(categorize,inplace=True)

X_test = pd.DataFrame(data=data_test, columns = ['statement'])
y_test = pd.DataFrame(data=data_test, columns = ['label'])

X_test = X_test.statement.tolist()
y_test = np.array(y_test.label)

# Bag-of-word (vectorizer)

#vectorizer = CountVectorizer(tokenizer=tokenize, lowercase = False)
vectorizer = CountVectorizer()
vectorizer.fit_transform(X_train)
print(len(vectorizer.vocabulary_))

X_train = vectorizer.transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

print(len(X_train[0]))
# # Naive Bayes Classifier
#
# clf = MultinomialNB()
# clf.fit(X_train, y_train)
# predicted = clf.predict(X_test)
# accuracy = np.mean(predicted == y_test)
# print('naive bayes:', accuracy)
#
# clf2 = svm.SVC(kernel='linear')
# clf2.fit(X_train, y_train)
# predicted = clf2.predict(X_test)
# accuracy = np.mean(predicted == y_test)
# print('svm:', accuracy)
