import string

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import svm

from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

def tokenize(document):
    lemmatizer = WordNetLemmatizer()

    "Break the document into sentences"
    for sent in sent_tokenize(document):

        "Break the sentence into part of speech tagged tokens"
        for token, tag in pos_tag(wordpunct_tokenize(sent)):

            "Apply preprocessing to the token"
            token = token.lower()  # Convert to lower case
            token = token.strip()  # Strip whitespace and other punctuations
            token = token.strip('_')  # remove _ if any
            token = token.strip('*')  # remove * if any

            "If stopword, ignore."
            if token in stopwords.words('english'):
                continue

            "If punctuation, ignore."
            if all(char in string.punctuation for char in token):
                continue

            "If number, ignore."
            if token.isdigit():
                continue

            # Lemmatize the token and yield
            # Note: Lemmatization is the process of looking up a single word form
            # from the variety of morphologic affixes that can be applied to
            # indicate tense, plurality, gender, etc.
            lemma = lemmatizer.lemmatize(token)
            yield lemma


train_data = pd.read_table('train.tsv')[['label', 'statement']]
test_data = pd.read_table('test.tsv')[['label', 'statement']]

categorize = {'label': {'mostly-true':'TRUE', 'half-true':'TRUE', 'barely-true':'FALSE', 'pants-fire':'FALSE'}}
train_data.replace(categorize,inplace=True)
test_data.replace(categorize,inplace=True)

X_train = train_data['statement']
y_train = train_data['label']

X_test = test_data['statement']
y_test = test_data['label']

vectorizer = TfidfVectorizer(tokenizer=tokenize, preprocessor=None, lowercase=False).fit(X_train)

X_train_tfidf = vectorizer.transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_tfidf.toarray(), y_train)
Y_pred = nb.predict(X_test_tfidf.toarray())

print('Multinomial NB')
print(str(round(accuracy_score(y_test, Y_pred), 3) * 100) + '%')

svm = svm.SVC(kernel='linear')
svm.fit(X_train_tfidf.toarray(), y_train)
Y_pred = svm.predict(X_test_tfidf.toarray())

print('SVM')
print(str(round(accuracy_score(y_test, Y_pred), 3) * 100) + '%')