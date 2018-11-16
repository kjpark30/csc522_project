import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from tokenizer import tokenize

train_data = pd.read_table('../train.tsv')[['label', 'statement']]
test_data = pd.read_table('../test.tsv')[['label', 'statement']]

data = pd.concat([train_data, test_data], ignore_index=True)

X = data['statement']
y_six = data['label']
y_two = data['label'].apply(lambda x: 'TRUE' if x in ['TRUE', 'mostly-true', 'half-true'] else 'FALSE')

print(y_six.value_counts())
print('')
print(y_two.value_counts())
print('')

tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False)
count_vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False)

X_tfidf = tfidf_vectorizer.fit_transform(X).toarray()
X_bow = count_vectorizer.fit_transform(X).toarray()

skfolds = StratifiedKFold(n_splits=10, random_state=0)

clf = SVC(kernel='rbf', gamma='scale')

accuracy_tfidf_six = []
accuracy_bow_six = []

for train_index, test_index in skfolds.split(X_tfidf, y_six):
    cloned_clf = clone(clf)

    X_train_folds = X_tfidf[train_index]
    y_train_folds = y_six[train_index]

    X_test_folds = X_tfidf[test_index]
    y_test_folds = y_six[test_index]

    cloned_clf.fit(X_train_folds, y_train_folds)

    y_pred = cloned_clf.predict(X_test_folds)

    accuracy_tfidf_six.append(accuracy_score(y_test_folds, y_pred))

for train_index, test_index in skfolds.split(X_bow, y_six):
    cloned_clf = clone(clf)

    X_train_folds = X_bow[train_index]
    y_train_folds = y_six[train_index]

    X_test_folds = X_bow[test_index]
    y_test_folds = y_six[test_index]

    cloned_clf.fit(X_train_folds, y_train_folds)

    y_pred = cloned_clf.predict(X_test_folds)

    accuracy_bow_six.append(accuracy_score(y_test_folds, y_pred))

tfidf_perf_six = pd.Series(accuracy_tfidf_six)
bow_perf_six = pd.Series(accuracy_bow_six)

print('tfidf_perf_six')
print(tfidf_perf_six.describe())
print('')
print('bow_perf_six')
print(bow_perf_six.describe())
print('')


accuracy_tfidf_two = []
accuracy_bow_two = []

for train_index, test_index in skfolds.split(X_tfidf, y_two):
    cloned_clf = clone(clf)

    X_train_folds = X_tfidf[train_index]
    y_train_folds = y_two[train_index]

    X_test_folds = X_tfidf[test_index]
    y_test_folds = y_two[test_index]

    cloned_clf.fit(X_train_folds, y_train_folds)

    y_pred = cloned_clf.predict(X_test_folds)

    accuracy_tfidf_two.append(accuracy_score(y_test_folds, y_pred))


for train_index, test_index in skfolds.split(X_bow, y_two):
    cloned_clf = clone(clf)

    X_train_folds = X_bow[train_index]
    y_train_folds = y_two[train_index]

    X_test_folds = X_bow[test_index]
    y_test_folds = y_two[test_index]

    cloned_clf.fit(X_train_folds, y_train_folds)

    y_pred = cloned_clf.predict(X_test_folds)

    accuracy_bow_two.append(accuracy_score(y_test_folds, y_pred))


tfidf_perf_two = pd.Series(accuracy_tfidf_two)
bow_perf_two = pd.Series(accuracy_bow_two)

print(print('tfidf_perf_two'))
print(tfidf_perf_two.describe())
print('')
print(print('bow_perf_two'))
print(bow_perf_two.describe())
print('')
