# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import shape
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import math

data = h5py.File('17signal15000.mat', 'r')
signals = data['signal'][:]
signals = np.transpose(signals)

signals_norm = np.linalg.norm(signals, axis=1, keepdims=True)
signals_norm_mean = np.mean(signals_norm)
print("mean norm of signals：", signals_norm_mean)

labels = np.zeros((15000, 1))
for i in range(1, 17):
    label = i * np.ones((15000, 1))
    labels = np.concatenate((labels, label))

pqd_type = ['Normal', 'Sag', 'Swell', 'Interruption', 'Transient/Impulse/Spike', 'Oscillatory transient', 'Harmonics',
            'Harmonics with Sag', 'Harmonics with Swell', 'Flicker', 'Flicker with Sag', 'Flicker with Swell',
            'Sag with Oscillatory transient', 'Swell with Oscillatory transient', 'Sag with Harmonics',
            'Swell with Harmonics', 'Notch']

np.random.seed(10)
index = np.arange(len(labels))
np.random.shuffle(index)

labels = labels[index]
teY_original = labels[230000:]
signals = signals[index]
signals = np.expand_dims(signals, axis=2)

trX = signals[:230000]
trY = labels[:230000]
teX = signals[230000:]
teY = labels[230000:]

universal_pert = np.load("20191014universal_pert_5000_1_2.0.npy")
print(np.linalg.norm(universal_pert, ord=2, keepdims=True))
length2 = 25000
adv_teX = teX[0:length2] + universal_pert
rea_teY = np.argmax(teY[0:length2], axis=1)

adv_univer = teX + universal_pert


########################################################calculate entropy###############################################
sns.set()
f, ax = plt.subplots()
C2 = np.load("confusion_matrix_SAA.npy")        ##### load other .npy files:confusion_matrix_SSA.npy/confusion_matrix_FGSM.npy

# C2 = np.ones([17,17])/17 ### max entropy

######prune:<0.05(1/17）
mask1 = C2<0.05
C2[mask1] = 0
######prune:<0.05(1/17）


def out_degree(C_matrix):
    ### weighted
    weighted_log_2 = list()
    for i in range(17):
        temp = 0
        for j in range(17):
            if C_matrix[i][j]!=0:
                temp = temp - C_matrix[i][j] * np.log2(C_matrix[i][j])
        weighted_log_2.append(temp)

    weighted_sum_2 = sum(weighted_log_2)

    ### unweighted
    C_matrix_unweighted = np.copy(C_matrix)
    C_matrix_unweighted[np.nonzero(C_matrix_unweighted)] = 1
    C_matrix_unweighted = C_matrix_unweighted / C_matrix_unweighted.sum(axis=1).reshape(17, 1)
    unweighted_log_2 = list()
    for i in range(17):
        temp = 0
        for j in range(17):
            if C_matrix_unweighted[i][j]!=0:
                temp = temp - C_matrix_unweighted[i][j] * np.log2(C_matrix_unweighted[i][j])
        unweighted_log_2.append(temp)

    unweighted_sum_2 = sum(unweighted_log_2)

    return np.array(weighted_log_2), np.array(weighted_sum_2), np.array(unweighted_log_2), np.array(unweighted_sum_2)

W_node_outentropy, W_graph_outentropy, UW_node_outentropy, UW_graph_outentropy = out_degree(C2)


####in-degree matrix
sum_column = C2.sum(axis=0).reshape(1,17)
sum_column[sum_column==0] = 1
C22 = C2 / sum_column

def in_degree(C_matrix):
    ### weighted
    weighted_log_2 = list()
    for j in range(17):
        if np.sum(C_matrix[:,j])== 0:
            weighted_log_2.append(0)
            continue
        temp = 0
        for i in range(17):
            if C_matrix[i][j]!=0:
                temp = temp - C_matrix[i][j] * np.log2(C_matrix[i][j])
        weighted_log_2.append(temp)

    weighted_sum_2 = sum(weighted_log_2)

    ### unweighted
    C_matrix_unweighted = np.copy(C_matrix)
    C_matrix_unweighted[np.nonzero(C_matrix_unweighted)] = 1
    sum_column = C_matrix_unweighted.sum(axis=0).reshape(1, 17)
    sum_column[sum_column == 0] = 1
    C_matrix_unweighted = C_matrix_unweighted / sum_column
    unweighted_log_2 = list()
    for j in range(17):
        if np.sum(C_matrix_unweighted[:,j])== 0:
            unweighted_log_2.append(0)
            continue
        temp = 0
        for i in range(17):
            if C_matrix_unweighted[i][j]!=0:
                temp = temp - C_matrix_unweighted[i][j] * np.log2(C_matrix_unweighted[i][j])
        unweighted_log_2.append(temp)

    unweighted_sum_2 = sum(unweighted_log_2)

    return np.array(weighted_log_2), np.array(weighted_sum_2), np.array(unweighted_log_2), np.array(unweighted_sum_2)

W_node_inentropy, W_graph_inentropy, UW_node_inentropy, UW_graph_inentropy = in_degree(C22)
########################################################calculate entropy###############################################

####################################################plot confusion_matrix###############################################
C2 = np.load("confusion_matrix_SAA.npy")
sns.heatmap(C2, annot=True, ax=ax, linewidths='0.5', cmap="BuPu",
            xticklabels=['C-1', 'C-2', 'C-3', 'C-4', 'C-5', 'C-6', 'C-7', 'C-8', 'C-9', 'C-10', 'C-11', 'C-12', 'C-13',
                         'C-14', 'C-15', 'C-16', 'C-17'],
            yticklabels=['C-1', 'C-2', 'C-3', 'C-4', 'C-5', 'C-6', 'C-7', 'C-8', 'C-9', 'C-10', 'C-11', 'C-12', 'C-13',
                         'C-14', 'C-15', 'C-16', 'C-17'])
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }
ax.set_title('Confusion matrix', font2)
ax.set_xlabel('Predict Class Label', font2)
ax.set_ylabel('True Label', font2)
# plt.subplots_adjust(left=0.05, right = 1.1, top=0.95, bottom=0.10)
plt.show()
####################################################plot confusion_matrix###############################################

print("done")