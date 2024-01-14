# SNN ECG classification for MIT-BIH dataset

## Dataset
The raw signal files (in .csv format) and annotation files are available for download from Kaggle at [kaggle.com/mondejar/mitbih-database](https://kaggle.com/mondejar/mitbih-database). We utilized data preprocessing scripts from [https://github.com/mondejar/ecg-classification](https://github.com/mondejar/ecg-classification) for processing the data.


## Training
We initially utilize 'training_testing_split.py' to segregate our dataset into training, validation, and testing sets. 
Subsequently, 'data_auge_binary.py' is employed to generate a binary dataset. We then execute 'python ecg_snn_2class.py --epochs 1 --lr 1' for preliminary identification of whether the signal is normal or abnormal.
For signals classified as abnormal, we further process them using 'data_auge_4class.py' followed by 'python ecg_snn_2class.py --epochs 5 --lr 1e-2' to categorize them into Normal, SVEB, VEB, or F categories. The training log is provided for reference.

## SNN accuracy
For 2-class SNN, the confusion matrix is:

 [[36255  7778]
 
 [ 1051  4607]]
Test set: Average loss: 1.4844, Accuracy: 40862/49691 (82%)

For 4-class SNN, the confusion matrix is:
Confusion Matrix:

 [[4926 1635 1195   22]
 
 [ 205 1098  126    0]
 
 [  22   33 2656   94]
 
 [  31  215  126    1]]
 
Test set: Average loss: 20.2795, Accuracy: 8681/12385 (70%)

The overall SNN accuracy is 90.43% with T=50.


