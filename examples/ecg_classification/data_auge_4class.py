import numpy as np
from collections import Counter

data = np.load("train_nor.npz")
train_data = data['data']
train_labels = data['label']

unique, counts = np.unique(train_labels, return_counts=True)
print("save",dict(zip(unique, counts)))

from imblearn.over_sampling import SMOTE
import numpy as np

class_counts = Counter(train_labels)

train_data_3 = []
train_data_label_3 = []
for i in range(len(train_labels)):
    if train_labels[i]==0:
        train_data_3.append(train_data[i])
        train_data_label_3.append(0)

    if train_labels[i]==3:
        for _ in range(90):
            train_data_3.append(train_data[i])
            train_data_label_3.append(3)

    if train_labels[i]==1:
        for _ in range(40):
            train_data_3.append(train_data[i])
            train_data_label_3.append(1)

    if train_labels[i]==2:
        for _ in range(10):
            train_data_3.append(train_data[i])
            train_data_label_3.append(2)

smote = SMOTE(k_neighbors=2,random_state=3407)
# 应用SMOTE
train_data_resampled, train_labels_resampled = smote.fit_resample(train_data_3, train_data_label_3)


np.savez('train_nor_4class.npz', data=train_data_resampled, label=train_labels_resampled)


# 检查新的类别分布
unique, counts = np.unique(train_labels_resampled, return_counts=True)
print("save",dict(zip(unique, counts)))
