# AT_SG:Adversarial Attacks and Defense Methods in the Smart Grid
This is the repository of the code which can be used for reproducing results using the algorithms and methods proposed in the paper "Adversarial Attacks and Defense Methods in the Smart Grid". 

Please cite the following paper, if you use the code provided in this repo:

Jiwei Tian, Buhong Wang,Tengyao Li, Fute Shang, Kunrui Cao, and Mete Ozay, Adversarial Attacks and Defense Methods in the Smart Grid, 2020.


## Table of Contents
0. [Introduction](#introduction)
0. [Training CNN Models using Datasets of Normal Signals](#Training-CNN-Models-using-Datasets-of-Normal-Signals)
0. [Attacks on the Trained CNN Models](#Attacks-on-the-Trained-CNN-Models)
0. [Defending CNN Models using Adversarial Training](#Defending-CNN-Models-using-Adversarial-Training)
0. [Qualitative Analyses and Visualization of Signals and Learned Feature Representations](#Qualitative-Analyses-and-Visualization-of-Signals-and-Learned-Feature-Representations)
0. [References](#References)



### Introduction

Vulnerability of various machine learning methods to adversarial examples has been recently explored in the literature. Power systems which use these vulnerable methods face a huge threat against adversarial examples. To this end, we first propose a more accurate signal-specific method and a universal signal-agnostic method to attack power systems using generated adversarial examples. We then adopt adversarial training to defend against attacks of adversarial examples.

### Training CNN Models using Datasets of Normal Signals

Note: In order to run the code, please put the data downloaded from github or dropbox together to the code folder.

A labeled dataset of 255000 signals was constructed using 15000 signals belonging to each class ([17signal15000.mat](https://www.dropbox.com/sh/aprts9x8l2frcjl/AABCuJ3TsJkSSLj2ZixeAyDAa?dl=0)).

The voltage signals are generated considering the following references [1,2,3].

The deep CNN model is inspired by the model proposed in [4] (Neural_Net_Module.py).

Power quality assessment by classifying (17 classes) voltage signals using deep CNN (pqd_classify.py).

The trained models achieves 92.05% +/- 0.11 and 92.01% +/- 0.11 training and test accuracy, respectively (one trained model:17oldweights_dnn_clean10.h5).

### Attacks on the Trained CNN Models

#### Signal-specific Attack Method (SSA)

pqd_SSA.py: Implementation of the proposed signal-specific attack method (SSA) to generate adversarial perturbation of signals.

For comparsion, we also implemented the FGSM based attack method (pqd_FGSM.py).

#### Signal-agnostic Attack Method (SAA)

pqd_SAA.py and SSA.py: Implementation of the proposed signal-agnostic attack method (SAA) to generate universal perturbations that make trained CNN models misclassify natural PQ signals with high probability.

universal_pert_5000_1_2.0.npy: generated universal perturbations. The overall misclassification probability can reach to 74% on the adversarial signals using the generated universal perturbations.

pqd_signal_all.py: Code for generating randomly chosen clean signals (original) and the corresponding adversarial signals generated using the SAA, FGSM and SAA.

### Defending CNN Models using Adversarial Training

In this section, we provide the code which can be used for implementing adversarial training as a defense method to improve robustness of learning models employed in power systems:

pqd_adv_training_FGSM.py: Code for adversarial training using adversarial signals generated by FGSM;

The output robustness_epoch_FGSM is saved to the file Perturbation_percent_FGSM_SSA_mean.npy

pqd_adv_training_SSA.py: Code for adversarial training using adversarial signals generated by SSA;

The output robustness_epoch_SSA is saved to the file Perturbation_percent_SSA_SSA_mean.npy

Code for training the model using the original dataset for a fair comparison (based on pqd_classify.py):

The output robustness_normal is saved to the file Perturbation_percent_normal_mean.npy

Code for analysis of robustness of the CNN model before adversarial training (based on pqd_SSA.py):

The output robustness_original is saved to the file Original_robustness.npy

compare_adv_training.py: Code used for comparison of adversarial training.

pqd_learning_curve.py: Code for displaying learning curves (data/8-trail).



### Qualitative Analyses and Visualization of Signals and Learned Feature Representations

plot_confusion_matrix.py: Code used for qualitatively analysis of the relationship between the characteristics of adversarial attacks to signals and defense methods, such as by calculating normalized confusion matrices and computation of class-confusion graphs (based on Gephi software).

pqd_visualization.py: Code used for visualizing signal measurements and their feature representations learned at different layers of DNNs using the t-SNE.

advtrain_weights_5.h5: The file containing optimized weights for adversarial training.

Please find the adverarial signals generated using the SSA in the following link: [adv_SSA_overshoot0.01_all.npy](https://www.dropbox.com/sh/aprts9x8l2frcjl/AABCuJ3TsJkSSLj2ZixeAyDAa?dl=0)


### References

[1] Igual, Raúl, et al. "Integral mathematical model of power quality disturbances." 2018 18th International Conference on Harmonics and Quality of Power (ICHQP). IEEE, 2018.

[2] Mohammadi, Mohammad, et al. "Detection and Classification of Multiple Power Quality Disturbances based on Temporal Deep Learning." 2019 IEEE International Conference on Environment and Electrical Engineering and 2019 IEEE Industrial and Commercial Power Systems Europe (EEEIC/I&CPS Europe). IEEE, 2019.

[3] Lee C Y, Shen Y X. Optimal feature selection for power-quality disturbances classification[J]. IEEE Transactions on power delivery, 2011, 26(4): 2342-2351.

[4] Wang S, Chen H. A novel deep learning method for the classification of power quality disturbances using deep convolutional neural network[J]. Applied energy, 2019, 235: 1126-1140.


