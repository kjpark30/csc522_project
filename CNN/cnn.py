import os
import pandas as pd
from gensim import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers
import os
from gensim.test.utils import datapath
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('/home/shwetha/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

from sklearn.metrics import accuracy_score

#TESTING DATA
tsv_file_test='test.tsv'
csv_table_test=pd.read_table(tsv_file_test,sep='\t')
target_test=csv_table_test['label']
words_input_test=csv_table_test['statement']


#TRAINING DATA
tsv_file_train='train.tsv'
csv_table_train=pd.read_table(tsv_file_train,sep='\t')
target_train=csv_table_train['label']
words_input_train=csv_table_train['statement']

category=csv_table_train.label.unique()
dic={}
for i,label in enumerate(category):
    dic[label]=i
labels=csv_table_train.label.apply(lambda x:dic[x])

val_data=csv_table_train.sample(frac=0.2,random_state=200)
train_data=csv_table_train.drop(val_data.index)

texts=train_data.statement


NUM_WORDS=20000
tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
tokenizer.fit_on_texts(texts)
sequences_train = tokenizer.texts_to_sequences(texts)
sequences_valid=tokenizer.texts_to_sequences(val_data.statement)
word_index = tokenizer.word_index

X_train = pad_sequences(sequences_train)
X_val = pad_sequences(sequences_valid,maxlen=X_train.shape[1])
y_train = to_categorical(np.asarray(labels[train_data.index]))
y_val = to_categorical(np.asarray(labels[val_data.index]))

EMBEDDING_DIM=300
vocabulary_size=min(len(word_index)+1,NUM_WORDS)
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
for word, i in word_index.items():
    if i>=NUM_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

del(word_vectors)

from keras.layers import Embedding
embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)


sequence_length = X_train.shape[1]
filter_sizes = [3,4,5]
num_filters = 100
drop = 0.5
# Just disables the warning, doesn't enable AVX/FMA

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


inputs = Input(shape=(sequence_length,))
embedding = embedding_layer(inputs)
reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)

conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)

maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)

merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
flatten = Flatten()(merged_tensor)
reshape = Reshape((3*num_filters,))(flatten)
dropout = Dropout(drop)(flatten)
output = Dense(units=6, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)

# this creates a model that includes
model = Model(inputs, output)

adam = Adam(lr=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])
callbacks = [EarlyStopping(monitor='val_loss')]
model.fit(X_train, y_train, batch_size=1000, epochs=5, verbose=1, validation_data=(X_val, y_val),
         callbacks=callbacks)
sequences_test=tokenizer.texts_to_sequences(csv_table_test.statement)
X_test = pad_sequences(sequences_test,maxlen=X_train.shape[1])
y_pred=model.predict(X_test)



to_submit=pd.DataFrame(index=csv_table_test.id,data={'FALSE':y_pred[:,dic['FALSE']],
                                                'TRUE':y_pred[:,dic['TRUE']],
                                                'half-true':y_pred[:,dic['half-true']],
                                                'barely-true': y_pred[:, dic['barely-true']],
                                                 'pants-fire': y_pred[:, dic['pants-fire']],
                                                     'mostly-true': y_pred[:, dic['mostly-true']]})
to_submit.to_csv('submit.csv')