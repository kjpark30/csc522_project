import pandas as pd
import numpy as np
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def vaderSenti(statements):
	anz = SentimentIntensityAnalyzer()
	scores_3d = []
	scores_1d = []
	
	for statement in statements:
		scores = []
		vs = anz.polarity_scores(statement)
		scores_3d.append([vs['neg'], vs['neu'], vs['pos']])
		scores_1d.append(vs['compound'])
		
	return scores_1d, scores_3d

if __name__ == '__main__':
	
	train = pd.read_csv('train.tsv', delimiter='\t')
	test = pd.read_csv('test.tsv', delimiter='\t')
	
	total = pd.concat([train, test], ignore_index=True)
	
	# replace label with numbers
	two_class = {'label': {'TRUE':0, 'true':0, 'mostly-true':0, 'half-true':0,
							'barely-true':1, 'pants-fire':1, 'FALSE':1, 'false':1}}
							
	# replace label with numbers
	six_class = {'label': {'TRUE':0, 'true':0, 'mostly-true':1, 'half-true':2,
							'barely-true':3, 'pants-fire':4, 'FALSE':5, 'false':5}}

	# split X and y
	X = pd.DataFrame(data=total, columns = ['statement'])
	y_six = pd.DataFrame(data=total, columns = ['label'])	
	y_two = pd.DataFrame(data=total, columns = ['label'])	
	
	y_six.replace(six_class, inplace=True)
	y_two.replace(two_class, inplace=True)
	
	# extract sentiment scores from statments
	statements = list(X.statement)
	sent1d, sent3d = vaderSenti(statements)
	
	# using only sent1d - compound score to determine the negative/neutral/positive of the statment
	sent_onehot = []
	for x in range(len(sent1d)):
		if sent1d[x] >= 0.05:
			sent_onehot.append([0,0,1])
		elif (sent1d[x] > -0.05) & (sent1d[x] < 0.05):
			sent_onehot.append([0,1,0])
		else:
			sent_onehot.append([1,0,0])
	
	# saving 3 dimensional sentiment scores
	sent_to_save = np.array(sent3d)
	# saving 3 dimensional sentimen one-hot encoding
	#sent_to_save = np.array(sent_onehot)
	
	# save
	with open('./data/sentiment.pkl', 'wb') as handle:
		pickle.dump(sent_to_save, handle, protocol=2)

	# statistics for sentiment of statements
	df0 = pd.DataFrame(sent_onehot, columns=['neg', 'neu', 'pos'])
	df1 = pd.DataFrame(sent3d, columns=['neg', 'neu', 'pos'])
	df2 = pd.DataFrame(sent1d, columns=['comp'])
	data = pd.concat([df0, df2, y_six], axis=1, join_axes=[df1.index])
	data_raw = pd.concat([df1, df2, y_six], axis=1, join_axes=[df1.index])
	
	fk_neg = []
	fk_neu = []
	fk_pos = []
	re_neg = []
	re_neu = []
	re_pos = []
	
	c0 = data[data.label == 0]
	c1 = data[data.label == 1]
	c2 = data[data.label == 2]
	c3 = data[data.label == 3]
	c4 = data[data.label == 4]
	c5 = data[data.label == 5]

	cr0 = data_raw[data_raw.label == 0]
	cr1 = data_raw[data_raw.label == 1]
	cr2 = data_raw[data_raw.label == 2]
	cr3 = data_raw[data_raw.label == 3]
	cr4 = data_raw[data_raw.label == 4]
	cr5 = data_raw[data_raw.label == 5]	
	
	f_neg = len(c0[c0.neg == 1].neg.tolist())
	f_neu = len(c0[c0.neu == 1].neu.tolist())
	f_pos = len(c0[c0.pos == 1].pos.tolist())
	r_neg = len(c5[c5.neg == 1].neg.tolist())
	r_neu = len(c5[c5.neu == 1].neu.tolist())
	r_pos = len(c5[c5.pos == 1].pos.tolist())
	
	print("The number of neg / neu / pos in FAKE news = {} / {} / {}".format(f_neg, f_neu, f_pos))
	print("The number of neg / neu / pos in REAL news = {} / {} / {}".format(r_neg, r_neu, r_pos))'
