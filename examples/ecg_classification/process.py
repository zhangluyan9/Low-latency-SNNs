import numpy as np
from collections import Counter

data = np.load("4class_abnormal.npz")
train_data = data['data']
train_labels = data['labels']

nor=0
sve=0
veb = 0
f = 0
for i in range(len(train_data)):

    if train_labels[i]==0:
        nor+=1
    if train_labels[i]==1:
        sve+=1
    if train_labels[i]==2:
        veb+=1
        #train_labels[i]=1
    if train_labels[i]==3:
        f+=1
        #train_labels[i]=1

print(nor,sve,veb,f)

print(train_labels)


