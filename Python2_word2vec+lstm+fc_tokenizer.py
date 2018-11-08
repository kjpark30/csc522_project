import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import string

from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.merge import concatenate
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn import metrics


def preprocessing(input):
    for index, row in input.iterrows():
        yield simple_preprocess(row['statement'], min_len=2, max_len=15)


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


def to_sequence(statements, model):
    index2word_set = set(model.wv.index2word)
    sequences = []

    for statement in statements:
        # print(statement)
        # print("len(statement)=",len(statement))
        seq = []

        for i in range(len(statement)):
            word = statement[i]
            # print("{}th iter - word = {}".format(i,word))

            if word in index2word_set:
                wordvec = model[word].tolist()
                seq.append(wordvec)
            # else:
            #     print(word, "is not in the dictionary")

        sequences.append(seq)

    return sequences


if __name__ == '__main__':

    train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
    test = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')

    #     replace label with numbers
    categorize = {'label': {'TRUE': 0, 'mostly-true': 1, 'half-true': 2, 'barely-true': 3, 'pants-fire': 4, 'FALSE': 5}}

    train.replace(categorize, inplace=True)
    test.replace(categorize, inplace=True)

    #     split X and y
    X_train = pd.DataFrame(data=train, columns=['statement'])
    y_train = pd.DataFrame(data=train, columns=['label'])
    X_test = pd.DataFrame(data=test, columns=['statement'])
    y_test = pd.DataFrame(data=test, columns=['label'])

    # X_train = train['statement']4rv

    # y_train = train['label']
    # X_test = test['statement']
    # y_test = test['label']
    # preprocess the statements - tokenize, transform in lower case, etc
    # for Word2Vec
    # doc_train = list(preprocessing(X_train))
    # doc_test = list(preprocessing(X_test))

    doc_train = []
    for x in X_train['statement'].values:
        doc_train.append([t for t in tokenize(x)])

    doc_test = []
    for x in X_test['statement'].values:
        doc_test.append([t for t in tokenize(x)])
    # print(doc_train)
    # doc_train = list(tokenize(X_train))
#     doc_test = list(tokenize(X_test))
# 
#     """
#     # for vaderSentiment (sentiment analysis)
#     statements_train = list(X_train.statement)
#     statements_test = list(X_test.statement)f
# 
#     sentiment1d_train, sentiment3d_train = vaderSenti(statements_train)
#     sentiment1d_test, sentiment3d_test = vaderSenti(statements_test)
#     """
# 
    model = Word2Vec.load("Liar_size=150_window=10_min=2_workers=10_sample=0.001_epoch=10")

    seq_train = to_sequence(doc_train, model)
    seq_test = to_sequence(doc_test, model)

    print("seq_train")
    print(len(seq_train))
    print(len(seq_train[0]))
    print(len(seq_train[0][0]))
    print("seq_test")
    print(len(seq_test))
    print(len(seq_test[0]))
    print(len(seq_test[0][0]))

    X_train = pad_sequences(seq_train, maxlen=17, dtype='float32', value=0.0)
    X_test = pad_sequences(seq_test, maxlen=17, dtype='float32', value=0.0)

    print("the shapes of X_train and X_test")
    print(X_train.shape)
    print(X_test.shape)

    y_train = np.array(y_train['label'])
    print(y_train[:10])
    # one hot encode
    y_train_onehot = to_categorical(y_train)
    print(y_train_onehot[:10])

    input_1 = Input(shape=(X_train.shape[1], X_train.shape[2]))
    model_1 = LSTM(64, recurrent_dropout=0.5)(input_1)
    model_out = Dense(6, activation='softmax')(model_1)
    model = Model(inputs=input_1, outputs=model_out)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_onehot, epochs=30, batch_size=100, verbose=1)

    # predict_prob: predicted prob. of each class => [[0.1, 0.7, 0.1, 0.05, 0.05], [...]]
    predict_prob = model.predict(X_test)
    # predict = index of class which have the max value.
    predict = np.argmax(predict_prob, axis=1)

    #test_actual = y_test
    #test_pred = y_pred
    #test_pred_prob = predict_prob

    # recall = recall_score(y_test, predict)
    #f1 = f1_score(y_test, predict)
    #fpr, tpr, thresholds = metrics.roc_curve(y_test, predict_prob, pos_label=1)
    #auc = metrics.auc(fpr, tpr)
    print(predict)
    acc = accuracy_score(y_test, predict)
    # prec = precision_score(y_test, predict)

    # print('Recall = {}'.format(recall))
    #print(f1)
    #print(auc)
    print('Accuracy = {}'.format(acc))
    # print('Precision = {}'.format(prec))
