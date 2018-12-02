import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score

from tokenizer import tokenize

import numpy as np

train_data = pd.read_table('../train.tsv')[['label', 'statement']]
test_data = pd.read_table('../test.tsv')[['label', 'statement']]

data = pd.concat([train_data, test_data], ignore_index=True)

X = data['statement']
y_six = data['label']
y_two = data['label'].apply(lambda x: 0 if x in ['TRUE', 'mostly-true', 'half-true'] else 1)

print(y_six.value_counts())
print('')
print(y_two.value_counts())
print('')

prep = []
for x in X.values:
    prep.append(' '.join([t for t in tokenize(x)]))

X_prep = pd.Series(prep)

tfidf_vectorizer = TfidfVectorizer(lowercase=False)
count_vectorizer = CountVectorizer(lowercase=False)

# X_tfidf = tfidf_vectorizer.fit_transform(X).toarray()
# X_bow = count_vectorizer.fit_transform(X).toarray()

# pd.DataFrame(X_tfidf).to_csv(path_or_buf='tfidf.csv')
# pd.DataFrame(X_bow).to_csv(path_or_buf='bow.csv')

# X_tfidf = np.array(pd.read_csv('tfidf.csv'))
# X_bow = np.array(pd.read_csv('bow.csv'))

# print('Shape of TFIDF', len(X_tfidf), len(X_tfidf[0]))
# print('Shape of BOW', len(X_bow), len(X_bow[0]))
# print('')

skfolds = StratifiedKFold(n_splits=5, random_state=0)

# clf = GaussianNB()
clf = MultinomialNB()

accuracy_tfidf_six = []
accuracy_bow_six = []
recall_tfidf_six = []
recall_bow_six = []
precision_tfidf_six = []
precision_bow_six = []

for train_index, test_index in skfolds.split(X_prep, y_six):
    cloned_clf = clone(clf)

    X_train = X_prep[train_index]
    y_train = y_six[train_index]

    X_test = X_prep[test_index]
    y_test = y_six[test_index]

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    cloned_clf.fit(X_train_tfidf, y_train)

    y_pred = cloned_clf.predict(X_test_tfidf)

    accuracy_tfidf_six.append(accuracy_score(y_test, y_pred))
    recall_tfidf_six.append(recall_score(y_test, y_pred, average='macro'))
    precision_tfidf_six.append(precision_score(y_test, y_pred, average='macro'))

for train_index, test_index in skfolds.split(X_prep, y_six):
    cloned_clf = clone(clf)

    X_train = X_prep[train_index]
    y_train = y_six[train_index]

    X_test = X_prep[test_index]
    y_test = y_six[test_index]

    X_train_bow = count_vectorizer.fit_transform(X_train)
    X_test_bow = count_vectorizer.transform(X_test)

    cloned_clf.fit(X_train_bow, y_train)

    y_pred = cloned_clf.predict(X_test_bow)

    accuracy_bow_six.append(accuracy_score(y_test, y_pred))
    recall_bow_six.append(recall_score(y_test, y_pred, average='macro'))
    precision_bow_six.append(precision_score(y_test, y_pred, average='macro'))

tfidf_six = pd.DataFrame(data={'accuracy': accuracy_tfidf_six,
                               'recall': recall_tfidf_six,
                               'precision': precision_tfidf_six})

bow_six = pd.DataFrame(data={'accuracy': accuracy_bow_six,
                             'recall': recall_bow_six,
                             'precision': precision_bow_six})

print(tfidf_six)
print('')
print(bow_six)
print('')

accuracy_tfidf_two = []
accuracy_bow_two = []
recall_tfidf_two = []
recall_bow_two = []
precision_tfidf_two = []
precision_bow_two = []

for train_index, test_index in skfolds.split(X_prep, y_two):
    cloned_clf = clone(clf)

    X_train = X_prep[train_index]
    y_train = y_two[train_index]

    X_test = X_prep[test_index]
    y_test = y_two[test_index]

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    cloned_clf.fit(X_train_tfidf, y_train)

    y_pred = cloned_clf.predict(X_test_tfidf)

    accuracy_tfidf_two.append(accuracy_score(y_test, y_pred))
    recall_tfidf_two.append(recall_score(y_test, y_pred))
    precision_tfidf_two.append(precision_score(y_test, y_pred))


for train_index, test_index in skfolds.split(X_prep, y_two):
    cloned_clf = clone(clf)

    X_train = X_prep[train_index]
    y_train = y_two[train_index]

    X_test = X_prep[test_index]
    y_test = y_two[test_index]

    X_train_bow = count_vectorizer.fit_transform(X_train)
    X_test_bow = count_vectorizer.transform(X_test)

    cloned_clf.fit(X_train_bow, y_train)

    y_pred = cloned_clf.predict(X_test_bow)

    accuracy_bow_two.append(accuracy_score(y_test, y_pred))
    recall_bow_two.append(recall_score(y_test, y_pred))
    precision_bow_two.append(precision_score(y_test, y_pred))


tfidf_two = pd.DataFrame(data={'accuracy': accuracy_tfidf_two,
                               'recall': recall_tfidf_two,
                               'precision': precision_tfidf_two})

bow_two = pd.DataFrame(data={'accuracy': accuracy_bow_two,
                             'recall': recall_bow_two,
                             'precision': precision_bow_two})

print(tfidf_two)
print('')
print(bow_two)
print('')