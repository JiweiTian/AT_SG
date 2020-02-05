# AT_SG:Adversarial Attacks and Defense Methods in the Smart Grid
This is the code example for attacking a normal Neural Networks in the Smart Grid with adversarial inputs. 

Please put the data from github or dropbox together to the code folder.

## Table of Contents
0. [Introduction](#introduction)
0. [Training the normal CNN](#Training-the-normal-CNN)
0. [Attack the trained CNN](#Attack-the-trained-CNN)
0. [Defend——Adversarial training](#Defend-Adversarial-training)
0. [Qualitatively analyses and Visualization](#Qualitatively-analyses-and-Visualization)
0. [References](#References)



### Introduction

Vulnerability of various machine learning methods to adversarial examples has been recently explored in the literature. Power systems which use these vulnerable methods face a huge threat against adversarial examples. To this end, we first propose a more accurate signal-specific method and a universal signal-agnostic method to attack power systems using generated adversarial examples. We then adopt adversarial training to defend against attacks of adversarial examples.

### Training the normal CNN

Power quality assessment by classifying (17 classes) voltage signals using deep CNN (pqd_classify.py).

A labeled dataset of 255000 signals was constructed using 15000 signals belonging to each class ([17signal15000.mat](https://www.dropbox.com/sh/aprts9x8l2frcjl/AABCuJ3TsJkSSLj2ZixeAyDAa?dl=0)).

The voltage signals are generated considering the following references [1,2,3].

The deep CNN model is inspired by the reference [4] (Neural_Net_Module.py).

The trained models achieves 92.05% +/- 0.11 and 92.01% +/- 0.11 training and test accuracy, respectively (one trained model:17oldweights_dnn_clean10.h5).

### Attack the trained CNN

#### Signal-specific attack (SSA)

Implementing the proposed signal-specific and more accurate algorithm to generate adversarial perturbation of signals (pqd_SSA.py):

For comparsion, we also implemented the FGSM based attack method (pqd_FGSM.py).

#### Signal-agnostic attack (SAA)
Implementing the proposed signal-agnostic attack algorithm to generate universal perturbations that cause natural PQ signals to be misclassified with high probability:

pqd_SAA.py-----call SSA.py

universal_pert_5000_1_2.0.npy :the overall misclassification probability can reach to 74%.

Randomly chosen clean signals (original) and the corresponding adversarial signals generated using the SAA,FGSM and SAA:pqd_signal_all.py

### Defend Adversarial training

Using adversarial training as a defense method to improve robustness of learning models employed in power systems:

Adversarial training based on FGSM: pqd_adv_training_FGSM.py

robustness_epoch_FGSM = np.load("Perturbation_percent_FGSM_SSA_mean.npy")

Adversarial training based on SSA: pqd_adv_training_SSA.py

robustness_epoch_SSA = np.load("Perturbation_percent_SSA_SSA_mean.npy")

robustness_original = np.load("Original_robustness.npy")

robustness_normal = np.load("Perturbation_percent_normal_mean.npy")

Comparsion of adversarial training:compare_adv_training.py

leraning curve: pqd_learning_curve.py (data/8-trail)



##Qualitatively analyses and Visualization

Qualitatively analyze the relationship between the characteristics of adversarial attacks to signals and defense methods: plot_confusion_matrix.py 

Use t-SNE algorithm  to visualize signal measurements and their feature representations learned at different layers of DNNs: pqd_visualization.py

used adversarial training weights:advtrain_weights_5.h5

used generated adverarial signals based on SSA: [adv_SSA_overshoot0.01_all.npy](https://www.dropbox.com/sh/aprts9x8l2frcjl/AABCuJ3TsJkSSLj2ZixeAyDAa?dl=0)



The datasat is shared at the link below：
https://www.dropbox.com/sh/aprts9x8l2frcjl/AABCuJ3TsJkSSLj2ZixeAyDAa?dl=0

### References

[1] Igual, Raúl, et al. "Integral mathematical model of power quality disturbances." 2018 18th International Conference on Harmonics and Quality of Power (ICHQP). IEEE, 2018.

[2] Mohammadi, Mohammad, et al. "Detection and Classification of Multiple Power Quality Disturbances based on Temporal Deep Learning." 2019 IEEE International Conference on Environment and Electrical Engineering and 2019 IEEE Industrial and Commercial Power Systems Europe (EEEIC/I&CPS Europe). IEEE, 2019.

[3] Lee C Y, Shen Y X. Optimal feature selection for power-quality disturbances classification[J]. IEEE Transactions on power delivery, 2011, 26(4): 2342-2351.

[4] Wang S, Chen H. A novel deep learning method for the classification of power quality disturbances using deep convolutional neural network[J]. Applied energy, 2019, 235: 1126-1140.


