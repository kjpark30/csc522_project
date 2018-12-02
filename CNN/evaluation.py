import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

submit=pd.read_csv('submit.csv')
predicted_id=submit.id

tsv_file_test='test.tsv'
csv_table_test=pd.read_table(tsv_file_test,sep='\t')
target_test=csv_table_test['label']
target_id=csv_table_test['id']
label_array=['FALSE','TRUE','half-true','barely-true','pants-fire','mostly-true']
def minimum(a, n):
    maxpos = a.index(max(a))
    return maxpos

# for (index,elem1,elem2,elem3,elem4,elem5,elem6) in submit:
#            a=[elem1,elem2,elem3,elem4,elem5,elem6]
#            pos=minimum(a,len(a))
#            X=np.append(index,label_array[pos+1])

X=[]
Y=[]
A=[]
B=[]
Z=[]
for index, row in submit.iterrows():
      a=[row['FALSE'],row['TRUE'],row['half-true'],row['barely-true'],row['pants-fire'],row['mostly-true']]
      pos = minimum(a, len(a))
      X.append([row['id'],label_array.__getitem__(pos)])
      Y.append(row['id'])
      B.append(label_array.__getitem__(pos))

for index,row in csv_table_test.iterrows():
    if row['id'] in Y:
        Z.append([row['id'],row['label']])
        A.append(row['label'])


print("Accuracy is",accuracy_score(A,B))