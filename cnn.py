import tsv
import tensorflow as tf
import vector as vector
import numpy as np
from gensim.models import Word2Vec, word2vec
import pandas as pd
from keras.layers import Dense, Input, Flatten, Dropout, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from gensim.utils import simple_preprocess
from keras.models import Sequential

from sklearn.metrics import accuracy_score

#TESTING DATA
tsv_file_test='test.tsv'
csv_table=pd.read_table(tsv_file_test,sep='\t')
target_test=csv_table['label']
words_input_test=csv_table['statement']


#TRAINING DATA
tsv_file_train='train.tsv'
csv_table_train=pd.read_table(tsv_file_train,sep='\t')
target_train=csv_table_train['label']
words_input_train=csv_table_train['statement']

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


doc_train = list(preprocessing(pd.DataFrame(words_input_train)))
doc_test = list(preprocessing(pd.DataFrame(words_input_test)))


def to_sequence(statements, model):

    index2word_set = set(model.wv.index2word)
    sequences = []

    for statement in statements:
        #print(statement)
        #print("len(statement)=",len(statement))
        seq = []

        for i in range(len(statement)):
            word = statement[i]
            #print("{}th iter - word = {}".format(i,word))

            if word in index2word_set:
                wordvec = model[word].tolist()
                seq.append(wordvec)
            else:
                 print(word,"is not in the dictionary")

        sequences.append(seq)

    return sequences



#
# doc_train =list(tokenize(words_input_train))
# doc_test = list(tokenize(words_input_test))

model=Word2Vec.load('word2vec')
seq_train = to_sequence(doc_train, model)
seq_test = to_sequence(doc_test, model)






EMBEDDING_DIM = 150 # how big is each word vector
MAX_VOCAB_SIZE = 17000 # how many unique words to use (i.e num rows in embedding vector)
MAX_SEQUENCE_LENGTH =11 # max number of words in a comment to use

#training params
batch_size = 256
num_epochs = 2


train_cnn_data = pad_sequences(seq_train, maxlen=15, dtype='float32', value=0.0)
test_cnn_data= pad_sequences(seq_test, maxlen=15, dtype='float32', value=0.0)
y_train=test_cnn_data

train_embedding_weights = np.zeros((len(seq_train)+1, EMBEDDING_DIM))
for word,index in words_input_train:
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)


test_cnn_data = pad_sequences(seq_test, maxlen=MAX_SEQUENCE_LENGTH)

def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, trainable=False, extra_conv=True):
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)


    convs = []
    filter_sizes = [3, 4, 5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = Concatenate(mode='concat', concat_axis=1)(convs)


    conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv)

    if extra_conv == True:
        x = Dropout(0.5)(l_merge)
    else:

        x = Dropout(0.5)(pool)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    # Finally, we feed the output into a Sigmoid layer.
    # The reason why sigmoid is used is because we are trying to achieve a binary classification(1,0)
    # for each of the 6 labels, and the sigmoid function will squash the output between the bounds of 0 and 1.
    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model

x_train = train_cnn_data
y_tr = y_train

#replace with word2vec
model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(seq_train)+1, EMBEDDING_DIM,
                len(list(target_train)), False)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]

hist = model.fit(x_train, y_tr, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, batch_size=batch_size)
y_test = model.predict(test_cnn_data, batch_size=1024, verbose=1)
print(accuracy_score(y_test,target_test))