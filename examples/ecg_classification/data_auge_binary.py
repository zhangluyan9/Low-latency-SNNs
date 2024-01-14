import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
####################################
#Training
data = np.load("train_nor.npz")
train_data = data['data']
train_labels = data['label']
unique, counts = np.unique(train_labels, return_counts=True)
print(dict(zip(unique, counts)))
train_data_3 = []
train_data_label_3 = []
for i in range(len(train_labels)):
    if train_labels[i]==0:
        train_data_3.append(train_data[i])
        train_data_label_3.append(0)
    if train_labels[i]!=0:
        train_data_3.append(train_data[i])
        train_data_label_3.append(1)

smote = SMOTE(k_neighbors=2,random_state=3407)
train_data_resampled, train_labels_resampled = smote.fit_resample(train_data_3, train_data_label_3)

unique, counts = np.unique(train_labels_resampled, return_counts=True)
print(dict(zip(unique, counts)))
np.savez('train_nor_binary.npz', data=train_data_resampled, label=train_labels_resampled)

####################################
#Testing
data = np.load("test_nor.npz")
train_data = data['data']
train_labels = data['label']
unique, counts = np.unique(train_labels, return_counts=True)
print(dict(zip(unique, counts)))

train_data_3 = []
train_data_label_3 = []
for i in range(len(train_labels)):
    if train_labels[i]==0:
        train_data_3.append(train_data[i])
        train_data_label_3.append(0)
    if train_labels[i]!=0:
        train_data_3.append(train_data[i])
        train_data_label_3.append(1)

train_data_3 = np.array(train_data_3)
train_data_label_3 = np.array(train_data_label_3)

np.savez('test_nor_binary.npz', data=train_data_3, label=train_data_label_3)

unique, counts = np.unique(train_data_label_3, return_counts=True)
print(dict(zip(unique, counts)))


####################################
#Val
data = np.load("val_nor.npz")
train_data = data['data']
train_labels = data['label']
unique, counts = np.unique(train_labels, return_counts=True)
print(dict(zip(unique, counts)))

train_data_3 = []
train_data_label_3 = []
for i in range(len(train_labels)):
    if train_labels[i]==0:
        train_data_3.append(train_data[i])
        train_data_label_3.append(0)
    if train_labels[i]!=0:
        train_data_3.append(train_data[i])
        train_data_label_3.append(1)

train_data_3 = np.array(train_data_3)
train_data_label_3 = np.array(train_data_label_3)

np.savez('val_nor_binary.npz', data=train_data_3, label=train_data_label_3)

unique, counts = np.unique(train_data_label_3, return_counts=True)
print(dict(zip(unique, counts)))

