import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
#from sklearn.cross_validation import KFold
from sklearn.model_selection import StratifiedKFold
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import pickle
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

np.random.seed(7)

def preprocessing(input):
	for index, row in input.iterrows():
		yield simple_preprocess(row['statement'], min_len=2, max_len=15)

def vaderSenti(statements):
	analyzer = SentimentIntensityAnalyzer()
	scores_3d = []
	scores_1d = []
	for statement in statements:
		scores = []
		vs = analyzer.polarity_scores(statement)
		#print("{:-<65} {}".format(statement, str(vs)))
		#scores.append()
		#print("neg={}, neu={}, pos={}, compound={}".format(vs['neg'], vs['neu'], vs['pos'], vs['compound']))
		scores_3d.append([vs['neg'], vs['neu'], vs['pos']])
		scores_1d.append(vs['compound'])
	
	return scores_1d, scores_3d
	
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
			#	 print(word, "is not in the dictionary")

		sequences.append(seq)

	return sequences


if __name__ == '__main__':

	train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
	test = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')
	#train = pd.read_csv('train.tsv', delimiter='\t')
	#test = pd.read_csv('test.tsv', delimiter='\t')

	total = pd.concat([train, test], ignore_index=True)
	#total.to_csv("total.csv", index=False)	
	
	# replace label with numbers
	two_class = {'label': {'TRUE':0, 'true':0, 'mostly-true':0, 'half-true':0,
							'barely-true':1, 'pants-fire':1, 'FALSE':1, 'false':1}}
							
	# replace label with numbers
	six_class = {'label': {'TRUE':0, 'true':0, 'mostly-true':1, 'half-true':2,
							'barely-true':3, 'pants-fire':4, 'FALSE':5, 'false':5}}

	# split X and y
	X = pd.DataFrame(data=total, columns = ['statement'])
	#y = pd.DataFrame(data=total, columns = ['label'])
	y_six = pd.DataFrame(data=total, columns = ['label'])	
	y_two = pd.DataFrame(data=total, columns = ['label'])	
	
	y_six.replace(six_class, inplace=True)
	y_two.replace(two_class, inplace=True)
	
	doc = []
	for x in X['statement'].values:
		doc.append([t for t in tokenize(x)])

	'''
	encoded_text = []
	statements = list(X.statement)	
	for text in statements:
		text = str([x.encode('utf-8') for x in text])
		encoded_text.append(text)

	encoded_text = np.asarray(encoded_text)

	sentiment1d, sentiment3d = vaderSenti(encoded_text)
	sentiment3d = np.asarray(sentiment3d)	 
	'''
	#sentiment3d = np.load('sentiment.pkl')
	
	with open('sentiment.pkl', 'rb') as f:
		sentiment3d = pickle.load(f)
	
	# custom model
	model_c = Word2Vec.load("Liar_size=150_window=10_min=2_workers=10_sample=0.001_epoch=10")
	# Google pre-trained model
	model_g = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

	temp_c = to_sequence(doc, model_c)
	temp_g = to_sequence(doc, model_g)
	
	# maxlen=17?
	X_custom = pad_sequences(temp_c, maxlen=20, dtype='float32', value=0.0)
	X_google = pad_sequences(temp_g, maxlen=20, dtype='float32', value=0.0)

	print("the shapes of X_custom")
	print(X_custom.shape)
	print("the shapes of X_google")
	print(X_google.shape)

	#y = np.array(y['label'])
	y_six = np.array(y_six['label'])
	y_two = np.array(y_two['label'])

	skf = StratifiedKFold(n_splits=10, random_state=0)
	
	acc_6cl_custom = []
	acc_6cl_custom_sentiment = []
	acc_2cl_custom = []
	acc_2cl_custom_sentiment = []
	acc_6cl_google = []
	acc_6cl_google_sentiment = []
	acc_2cl_google = []
	acc_2cl_google_sentiment = []

	recall_6cl_custom = []
	recall_6cl_custom_sentiment = []
	recall_2cl_custom = []
	recall_2cl_custom_sentiment = []
	recall_6cl_google = []
	recall_6cl_google_sentiment = []
	recall_2cl_google = []
	recall_2cl_google_sentiment = []
	
	prec_6cl_custom = []
	prec_6cl_custom_sentiment = []
	prec_2cl_custom = []
	prec_2cl_custom_sentiment = []
	prec_6cl_google = []
	prec_6cl_google_sentiment = []
	prec_2cl_google = []
	prec_2cl_google_sentiment = []
	
	
	# [6 class + custom W2V] ----------------------------------------------------------------
	for train_index, test_index in skf.split(X_custom, y_six):
		X_train, X_test = X_custom[train_index], X_custom[test_index]
		y_train, y_test = y_six[train_index], y_six[test_index]

		#one hot encoding
		y_train_onehot = to_categorical(y_train)
		print(y_train_onehot[:10])

		input_1 = Input(shape=(X_train.shape[1], X_train.shape[2]))
		model_1 = LSTM(64, recurrent_dropout=0.5)(input_1)
		model_out = Dense(6, activation='softmax')(model_1)
		model = Model(inputs=input_1, outputs=model_out)

		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(X_train, y_train_onehot, epochs=10, batch_size=100, verbose=1)

		# predict_prob: predicted prob. of each class => [[0.1, 0.7, 0.1, 0.05, 0.05], [...]]
		predict_prob = model.predict(X_test)
		# predict = index of class which have the max value.
		predict = np.argmax(predict_prob, axis=1)

		acc = accuracy_score(y_test, predict)	
		print(acc)
		acc_6cl_custom.append(acc)
		
		prec = precision_score(y_test, predict, average='macro')
		print(prec)
		prec_6cl_custom.append(prec)

		recall = recall_score(y_test, predict, average='macro')
		print(recall)
		recall_6cl_custom.append(recall)


	print("1) 6 class, custom w2v")
	print("Accuracy >>")
	print(acc_6cl_custom)
	acc_6cl_custom = pd.Series(acc_6cl_custom)
	print(acc_6cl_custom.describe())

	print("Precision >>")
	print(prec_6cl_custom)
	prec_6cl_custom = pd.Series(prec_6cl_custom)
	print(prec_6cl_custom.describe())

	print("Recall >>")
	print(recall_6cl_custom)
	recall_6cl_custom = pd.Series(recall_6cl_custom)
	print(recall_6cl_custom.describe())
	
	
	# [6 class + custom W2V + sentiment] ----------------------------------------------------------------
	for train_index, test_index in skf.split(X_custom, y_six):
		X_train, X_test = X_custom[train_index], X_custom[test_index]
		y_train, y_test = y_six[train_index], y_six[test_index]
		sentiment_train, sentiment_test = sentiment3d[train_index], sentiment3d[test_index]

		#one hot encoding
		y_train_onehot = to_categorical(y_train)
		print(y_train_onehot[:10])

		input_1 = Input(shape=(X_train.shape[1], X_train.shape[2]))
		model_1 = LSTM(64, recurrent_dropout=0.5)(input_1)

		input_2 = Input(shape=(sentiment_train.shape[1],))
		model_2 = Dense(64)(input_2)

		merged = concatenate([model_1, model_2], axis=1)
		merged_out = Dense(6, activation='softmax')(merged)
		model = Model(inputs=[input_1, input_2], outputs=merged_out)

		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit([X_train, sentiment_train], y_train_onehot, epochs=10, batch_size=100, verbose=1)

		# predict_prob: predicted prob. of each class => [[0.1, 0.7, 0.1, 0.05, 0.05], [...]]
		predict_prob = model.predict([X_test, sentiment_test])
		# predict = index of class which have the max value.
		predict = np.argmax(predict_prob, axis=1)

		acc = accuracy_score(y_test, predict)
		print(acc)
		acc_6cl_custom_sentiment.append(acc)
		
		prec = precision_score(y_test, predict, average='macro')
		print(prec)
		prec_6cl_custom_sentiment.append(prec)

		recall = recall_score(y_test, predict, average='macro')
		print(recall)
		recall_6cl_custom_sentiment.append(recall)
		
		

	print("2) 6 class, custom w2v + sentiment")
	print("Accuracy >>")
	print(acc_6cl_custom_sentiment)
	acc_6cl_custom_sentiment = pd.Series(acc_6cl_custom_sentiment)
	print(acc_6cl_custom_sentiment.describe())
	
	print("Precision >>")
	print(prec_6cl_custom_sentiment)
	prec_6cl_custom_sentiment = pd.Series(prec_6cl_custom_sentiment)
	print(prec_6cl_custom_sentiment.describe())

	print("Recall >>")
	print(recall_6cl_custom_sentiment)
	recall_6cl_custom_sentiment = pd.Series(recall_6cl_custom_sentiment)
	print(recall_6cl_custom_sentiment.describe())
	

	# [6 class + google W2V] ----------------------------------------------------------------
	for train_index, test_index in skf.split(X_google, y_six):
		X_train, X_test = X_google[train_index], X_google[test_index]
		y_train, y_test = y_six[train_index], y_six[test_index]

		#one hot encoding
		y_train_onehot = to_categorical(y_train)
		print(y_train_onehot[:10])

		input_1 = Input(shape=(X_train.shape[1], X_train.shape[2]))
		model_1 = LSTM(64, recurrent_dropout=0.5)(input_1)
		model_out = Dense(6, activation='softmax')(model_1)
		model = Model(inputs=input_1, outputs=model_out)

		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(X_train, y_train_onehot, epochs=10, batch_size=100, verbose=1)

		# predict_prob: predicted prob. of each class => [[0.1, 0.7, 0.1, 0.05, 0.05], [...]]
		predict_prob = model.predict(X_test)
		# predict = index of class which have the max value.
		predict = np.argmax(predict_prob, axis=1)

		acc = accuracy_score(y_test, predict)
		print(acc)
		acc_6cl_google.append(acc)
		
		prec = precision_score(y_test, predict, average='macro')
		print(prec)
		prec_6cl_google.append(prec)

		recall = recall_score(y_test, predict, average='macro')
		print(recall)
		recall_6cl_google.append(recall)

	print("3) 6 class, google w2v")
	print("Accuracy >>")
	print(acc_6cl_google)
	acc_6cl_google = pd.Series(acc_6cl_google)
	print(acc_6cl_google.describe())
	
	print("Precision >>")
	print(prec_6cl_google)
	prec_6cl_google = pd.Series(prec_6cl_google)
	print(prec_6cl_google.describe())

	print("Recall >>")
	print(recall_6cl_google)
	recall_6cl_google = pd.Series(recall_6cl_google)
	print(recall_6cl_google.describe())
	

	# [6 class + google W2V + sentiment] ----------------------------------------------------------------
	for train_index, test_index in skf.split(X_google, y_six):
		X_train, X_test = X_google[train_index], X_google[test_index]
		y_train, y_test = y_six[train_index], y_six[test_index]
		sentiment_train, sentiment_test = sentiment3d[train_index], sentiment3d[test_index]

		#one hot encoding
		y_train_onehot = to_categorical(y_train)
		print(y_train_onehot[:10])

		input_1 = Input(shape=(X_train.shape[1], X_train.shape[2]))
		model_1 = LSTM(64, recurrent_dropout=0.5)(input_1)

		input_2 = Input(shape=(sentiment_train.shape[1],))
		model_2 = Dense(64)(input_2)

		merged = concatenate([model_1, model_2], axis=1)
		merged_out = Dense(6, activation='softmax')(merged)
		model = Model(inputs=[input_1, input_2], outputs=merged_out)

		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit([X_train, sentiment_train], y_train_onehot, epochs=10, batch_size=100, verbose=1)

		# predict_prob: predicted prob. of each class => [[0.1, 0.7, 0.1, 0.05, 0.05], [...]]
		predict_prob = model.predict([X_test, sentiment_test])
		# predict = index of class which have the max value.
		predict = np.argmax(predict_prob, axis=1)

		acc = accuracy_score(y_test, predict)
		print(acc)
		acc_6cl_google_sentiment.append(acc)
		
		prec = precision_score(y_test, predict, average='macro')
		print(prec)
		prec_6cl_google_sentiment.append(prec)

		recall = recall_score(y_test, predict, average='macro')
		print(recall)
		recall_6cl_google_sentiment.append(recall)
		

	print("6) 6 class, google w2v + sentiment")
	print("Accuracy >>")
	print(acc_6cl_google_sentiment)
	acc_6cl_google_sentiment = pd.Series(acc_6cl_google_sentiment)
	print(acc_6cl_google_sentiment.describe())
	
	print("Precision >>")
	print(prec_6cl_google_sentiment)
	prec_6cl_google_sentiment = pd.Series(prec_6cl_google_sentiment)
	print(prec_6cl_google_sentiment.describe())

	print("Recall >>")
	print(recall_6cl_google_sentiment)
	recall_6cl_google_sentiment = pd.Series(recall_6cl_google_sentiment)
	print(recall_6cl_google_sentiment.describe())

	# [2 class + custom W2V] ----------------------------------------------------------------
	for train_index, test_index in skf.split(X_custom, y_two):
		X_train, X_test = X_custom[train_index], X_custom[test_index]
		y_train, y_test = y_two[train_index], y_two[test_index]

		input_1 = Input(shape=(X_train.shape[1], X_train.shape[2]))
		model_1 = LSTM(64, recurrent_dropout=0.5)(input_1)
		model_out = Dense(1, activation='sigmoid')(model_1)
		model = Model(inputs=input_1, outputs=model_out)

		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		model.fit(X_train, y_train, epochs=10, batch_size=100, verbose=1)

		# predict_prob: predicted prob. of each class => [[0.1, 0.7, 0.1, 0.05, 0.05], [...]]
		predict_prob = model.predict(X_test)
		predict = np.round(predict_prob)

		acc = accuracy_score(y_test, predict)
		print(acc)
		acc_2cl_custom.append(acc)

		prec = precision_score(y_test, predict)
		print(prec)
		prec_2cl_custom.append(prec)

		recall = recall_score(y_test, predict)
		print(recall)
		recall_2cl_custom.append(recall)

	print("2) 2 class, custom w2v")
	print("Accuracy >>")
	print(acc_2cl_custom)
	acc_2cl_custom = pd.Series(acc_2cl_custom)
	print(acc_2cl_custom.describe())

	print("Precision >>")
	print(prec_2cl_custom)
	prec_2cl_custom = pd.Series(prec_2cl_custom)
	print(prec_2cl_custom.describe())

	print("Recall >>")
	print(recall_2cl_custom)
	recall_2cl_custom = pd.Series(recall_2cl_custom)
	print(recall_2cl_custom.describe())

	# [2 class + custom W2V + sentiment] ----------------------------------------------------------------
	for train_index, test_index in skf.split(X_custom, y_two):
		X_train, X_test = X_custom[train_index], X_custom[test_index]
		y_train, y_test = y_two[train_index], y_two[test_index]
		sentiment_train, sentiment_test = sentiment3d[train_index], sentiment3d[test_index]

		input_1 = Input(shape=(X_train.shape[1], X_train.shape[2]))
		model_1 = LSTM(64, recurrent_dropout=0.5)(input_1)

		input_2 = Input(shape=(sentiment_train.shape[1],))
		model_2 = Dense(64)(input_2)

		merged = concatenate([model_1, model_2], axis=1)
		merged_out = Dense(1, activation='sigmoid')(merged)
		model = Model(inputs=[input_1, input_2], outputs=merged_out)

		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		model.fit([X_train, sentiment_train], y_train, epochs=10, batch_size=100, verbose=1)

		# predict_prob: predicted prob. of each class => [[0.1, 0.7, 0.1, 0.05, 0.05], [...]]
		predict_prob = model.predict([X_test, sentiment_test])
		predict = np.round(predict_prob)

		acc = accuracy_score(y_test, predict)
		print(acc)
		acc_2cl_custom_sentiment.append(acc)

		prec = precision_score(y_test, predict)
		print(prec)
		prec_2cl_custom_sentiment.append(prec)

		recall = recall_score(y_test, predict)
		print(recall)
		recall_2cl_custom_sentiment.append(recall)

	print("7) 2 class, custom w2v + sentiment")
	print("Accuracy >>")
	print(acc_2cl_custom_sentiment)
	acc_2cl_custom_sentiment = pd.Series(acc_2cl_custom_sentiment)
	print(acc_2cl_custom_sentiment.describe())

	print("Precision >>")
	print(prec_2cl_custom_sentiment)
	prec_2cl_custom_sentiment = pd.Series(prec_2cl_custom_sentiment)
	print(prec_2cl_custom_sentiment.describe())

	print("Recall >>")
	print(recall_2cl_custom_sentiment)
	recall_2cl_custom_sentiment = pd.Series(recall_2cl_custom_sentiment)
	print(recall_2cl_custom_sentiment.describe())

	# [2 class + google W2V] ----------------------------------------------------------------
	for train_index, test_index in skf.split(X_google, y_two):
		X_train, X_test = X_google[train_index], X_google[test_index]
		y_train, y_test = y_two[train_index], y_two[test_index]

		input_1 = Input(shape=(X_train.shape[1], X_train.shape[2]))
		model_1 = LSTM(64, recurrent_dropout=0.5)(input_1)
		model_out = Dense(1, activation='sigmoid')(model_1)
		model = Model(inputs=input_1, outputs=model_out)

		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		model.fit(X_train, y_train, epochs=10, batch_size=100, verbose=1)

		# predict_prob: predicted prob. of each class => [[0.1, 0.7, 0.1, 0.05, 0.05], [...]]
		predict_prob = model.predict(X_test)
		predict = np.round(predict_prob)

		acc = accuracy_score(y_test, predict)
		print(acc)
		acc_2cl_google.append(acc)

		prec = precision_score(y_test, predict)
		print(prec)
		prec_2cl_google.append(prec)

		recall = recall_score(y_test, predict)
		print(recall)
		recall_2cl_google.append(recall)

	print("4) 2 class, google w2v")
	print("Accuracy >>")
	print(acc_2cl_google)
	acc_2cl_google = pd.Series(acc_2cl_google)
	print(acc_2cl_google.describe())

	print("Precision >>")
	print(prec_2cl_google)
	prec_2cl_google = pd.Series(prec_2cl_google)
	print(prec_2cl_google.describe())

	print("Recall >>")
	print(recall_2cl_google)
	recall_2cl_google = pd.Series(recall_2cl_google)
	print(recall_2cl_google.describe())

	# [2 class + google W2V + sentiment] ----------------------------------------------------------------
	for train_index, test_index in skf.split(X_google, y_two):
		X_train, X_test = X_google[train_index], X_google[test_index]
		y_train, y_test = y_two[train_index], y_two[test_index]
		sentiment_train, sentiment_test = sentiment3d[train_index], sentiment3d[test_index]

		input_1 = Input(shape=(X_train.shape[1], X_train.shape[2]))
		model_1 = LSTM(64, recurrent_dropout=0.5)(input_1)

		input_2 = Input(shape=(sentiment_train.shape[1],))
		model_2 = Dense(64)(input_2)

		merged = concatenate([model_1, model_2], axis=1)
		merged_out = Dense(1, activation='sigmoid')(merged)
		model = Model(inputs=[input_1, input_2], outputs=merged_out)

		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		model.fit([X_train, sentiment_train], y_train, epochs=10, batch_size=100, verbose=1)

		# predict_prob: predicted prob. of each class => [[0.1, 0.7, 0.1, 0.05, 0.05], [...]]
		predict_prob = model.predict([X_test, sentiment_test])
		predict = np.round(predict_prob)

		acc = accuracy_score(y_test, predict)
		print(acc)
		acc_2cl_google_sentiment.append(acc)

		prec = precision_score(y_test, predict)
		print(prec)
		prec_2cl_google_sentiment.append(prec)

		recall = recall_score(y_test, predict)
		print(recall)
		recall_2cl_google_sentiment.append(recall)


	# ----------------------------------------
	print("8) 2 class, google w2v + sentiment")	
	print("Accuracy >>")
	print(acc_2cl_google_sentiment)
	acc_2cl_google_sentiment = pd.Series(acc_2cl_google_sentiment)
	print(acc_2cl_google_sentiment.describe())

	print("Precision >>")
	print(prec_2cl_google_sentiment)
	prec_2cl_google_sentiment = pd.Series(prec_2cl_google_sentiment)
	print(prec_2cl_google_sentiment.describe())

	print("Recall >>")
	print(recall_2cl_google_sentiment)
	recall_2cl_google_sentiment = pd.Series(recall_2cl_google_sentiment)
	print(recall_2cl_google_sentiment.describe())
  
