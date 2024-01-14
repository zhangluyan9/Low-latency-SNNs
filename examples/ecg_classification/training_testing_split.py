
import numpy as np
import os
import csv
import operator
from scipy.signal import medfilt
#DS1 = [101, 106]

#DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]

#DS1 = [108,114,118,124]
DS1_train = [101, 106, 109, 112, 115, 116, 119, 122, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS1_val = [108,114,118,124]
DS1_test = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

def create_dataset(DS1,name):
    #print(DS1,name)
    class_ID = [[] for i in range(len(DS1))]
    beat = [[] for i in range(len(DS1))] # record, beat, lead
    R_poses = [ np.array([]) for i in range(len(DS1))]
    Original_R_poses = [ np.array([]) for i in range(len(DS1))]   
    valid_R = [ np.array([]) for i in range(len(DS1))]
    #print(Original_R_poses)
    pathDB = '../data_kaggle/mit-bih-data/'

    fRecords = list()
    fAnnotations = list()
    size_RR_max = 20
    winL = 90
    winR= 90
    lst = os.listdir(pathDB)
    lst.sort()
    for file in lst:
        #print(file)
        if file.endswith(".csv"):
            if int(file[0:3]) in DS1:
                fRecords.append(file)
        elif file.endswith(".txt"):
            if int(file[0:3]) in DS1:
                fAnnotations.append(file)     

    MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']#, 'P', '/', 'f', 'u']
    AAMI_classes = []
    AAMI_classes.append(['N', 'L', 'R'])                    # N
    AAMI_classes.append(['A', 'a', 'J', 'S', 'e', 'j'])     # SVEB 
    AAMI_classes.append(['V', 'E'])                         # VEB
    AAMI_classes.append(['F'])                              # F
    #AAMI_classes.append(['P', '/', 'f', 'u'])              # Q

    RAW_signals = []
    r_index = 0

    for r in range(0, len(fRecords)):
        print("Processing signal " + str(r) + " / " + str(len(fRecords)) + "...")

        # 1. Read signalR_poses
        filename = pathDB + fRecords[r]
        f = open(filename, 'r')
        reader = csv.reader(f, delimiter=',')
        #first_line = next(reader)
        next(reader) # skip first line!

        MLII_index = 1
        if int(fRecords[r][0:3]) == 114:
            MLII_index = 2

        MLII = []
        for row in reader:
            MLII.append((int(row[MLII_index])))
        f.close()
        RAW_signals.append((MLII))

        # 2. Read annotations
        filename = pathDB + fAnnotations[r]
        f = open(filename, 'r')
        next(f) # skip first line!
        annotations = []
        for line in f:
            annotations.append(line)
        f.close
        #print(annotations[0:3])

        # 3. Preprocessing signal!

        baseline = medfilt(MLII, 71) 
        baseline = medfilt(baseline, 215) 

        # Remove Baseline
        for i in range(0, len(MLII)):
            MLII[i] = MLII[i] - baseline[i]

        for a in annotations:
            aS = a.split()
            pos = int(aS[1])
            originalPos = int(aS[1])
            classAnttd = aS[2]
            #print(pos,classAnttd)
            if pos > size_RR_max and pos < (len(MLII) - size_RR_max):
                index, value = max(enumerate(MLII[pos - size_RR_max : pos + size_RR_max]), key=operator.itemgetter(1))
                pos = (pos - size_RR_max) + index
            #print(pos)
            peak_type = 0
            #pos = pos-1
            
            if classAnttd in MITBIH_classes:
                if(pos > winL and pos < (len(MLII) - winR)):
                    beat[r].append( (MLII[pos - winL : pos + winR]))
                    for i in range(0,len(AAMI_classes)):
                        if classAnttd in AAMI_classes[i]:
                            class_AAMI = i
                            break #exit loop
                    #convert class
                    class_ID[r].append(class_AAMI)

                    valid_R[r] = np.append(valid_R[r], 1)
                else:
                    valid_R[r] = np.append(valid_R[r], 0)
            else:
                valid_R[r] = np.append(valid_R[r], 0)

            
            R_poses[r] = np.append(R_poses[r], pos)
            Original_R_poses[r] = np.append(Original_R_poses[r], originalPos)
    #beat = np.array(beat)
    #class_ID = np.array(class_ID)

    data = []
    label = []

    for i in range(len(beat)):
        beat_min = np.min(beat[i])
        beat_max = np.max(beat[i])
        normalized_beat = (beat[i] - beat_min) / (beat_max - beat_min)
        data.extend(normalized_beat)
        label.extend(class_ID[i])

    # 统计标签
    nor, sve, veb, f = np.bincount(label, minlength=4)
    print(nor, sve, veb, f)

    data = np.array(data)
    label = np.array(label)
    print(data.shape, label.shape)
    
    np.savez(name, data=data, label=label)

create_dataset(DS1_train, 'train_nor.npz')
create_dataset(DS1_val, 'val_nor.npz')
create_dataset(DS1_test, 'test_nor.npz')