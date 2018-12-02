import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score

from tokenizer import tokenize


def prep_data():
    train_data = pd.read_table('../train.tsv')[['label', 'statement']]
    test_data = pd.read_table('../test.tsv')[['label', 'statement']]

    data = pd.concat([train_data, test_data], ignore_index=True)

    X = data['statement']
    y_six = data['label']
    y_two = data['label'].apply(lambda x: 'TRUE' if x in ['TRUE', 'mostly-true', 'half-true'] else 'FALSE')

    print_vc(y_six)
    print_vc(y_two)

    y_two = y_two.apply(lambda x: 0 if x is 'TRUE' else 1)

    prep = []
    for x in X.values:
        prep.append(' '.join([t for t in tokenize(x)]))

    X_prep = pd.Series(prep)

    return X_prep, y_six, y_two


def nb_helper(X_prep, y_six, y_two, classifier, n_folds, desc):
    tfidf_vectorizer = TfidfVectorizer(lowercase=False)
    count_vectorizer = CountVectorizer(lowercase=False)

    classify(X_prep, y_six, classifier, tfidf_vectorizer, n_folds, 6, desc + '_TFiDF')
    classify(X_prep, y_six, classifier, count_vectorizer, n_folds, 6, desc + '_BoW')
    classify(X_prep, y_two, classifier, tfidf_vectorizer, n_folds, 2, desc + '_TFiDF')
    classify(X_prep, y_two, classifier, count_vectorizer, n_folds, 2, desc + '_BoW')


def classify(X, y, clf, vec, n_folds, n_class, operation='_'):
    skfolds = StratifiedKFold(n_splits=n_folds, random_state=100)

    accuracy = []
    recall = []
    precision = []

    for train_index, test_index in skfolds.split(X, y):
        cloned_clf = clone(clf)

        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        X_train_vec = vec.fit_transform(X_train).toarray()
        X_test_vec = vec.transform(X_test).toarray()

        cloned_clf.fit(X_train_vec, y_train)

        y_pred = cloned_clf.predict(X_test_vec)

        accuracy.append(accuracy_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred, average='binary' if n_class == 2 else 'macro'))
        precision.append(precision_score(y_test, y_pred, average='binary' if n_class == 2 else 'macro'))

    results = pd.DataFrame(data={'accuracy': accuracy, 'recall': recall, 'precision': precision})

    print('{} Fold CV Results for {} with {} Class Labels:'.format(n_folds, operation, n_class))
    print('')
    print(results.describe())
    print('')
    print('-----')
    print('\t')


def print_vc(y):
    vc = y.value_counts()
    n_class = len(vc)
    print('{} Class Labels:'.format(n_class))
    print('')
    for i in vc.index:
        print('{}: {}'.format(i, vc[i]))
    print('')
    print('-----')
    print('\t')


if __name__ == '__main__':
    print('\t')
    X_prep, y_six, y_two = prep_data()

    # TODO: Make changes below and run the file!
    # ...
    #
    # @Param: classifier => Any SciKit-Learn Classifier
    # @Param: n_folds => Number of Folds
    # @Param: desc => Name of Classifier
    #
    # ...
    nb_helper(X_prep, y_six, y_two, classifier=GaussianNB(), n_folds=10, desc='Gaussian_NB')
    nb_helper(X_prep, y_six, y_two, classifier=MultinomialNB(), n_folds=10, desc='Multinomial_NB')
