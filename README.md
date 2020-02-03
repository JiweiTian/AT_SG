# AT_SG
 Adversarial-Attacks-and-Defense-Methods-in-the-Smart-Grid
This is the code example for attacking a normal Neural Networks with adversarial inputs （Adversarial Attacks and Defense Methods in the Smart Grid). 

Please put the data from github or dropbox together to the code folder.

1> power quality assessment by classifying (17) voltage signals using deep CNN (pqd_classify.py).

A labeled dataset of 255000 signals was constructed using 15000 signals belonging to each class (17signal15000.mat).

the voltage signals are generated considering the following references:

"Integral mathematical model of power quality disturbances"

"Detection and Classification of Multiple Power Quality Disturbances based on Temporal Deep Learning"

"Optimal Feature Selection for Power-Quality Disturbances Classification"

the deep CNN model is inspired by the reference:

"A novel deep learning method for the classification of power quality disturbances using deep convolutional neural network" (Neural_Net_Module.py).

The trained model achieves 98.5% testing accuracy (17oldweights_dnn_clean10.h5).

2> Implementing the proposed signal-specific and more accurate algorithm to generate adversarial perturbation of signals (pqd_signal_specific_attack.py):

For comparsion, we also implemented the FGSM based attack method (pqd_FGSM.py).


3> Implementing the proposed signal-agnostic attack algorithm to generate universal perturbations that cause natural PQ signals to be misclassified with high probability:

（pqd_signal_agnostic_attack.py）

call signal_specific.py

universal_pert_5000_1_2.0.npy

the overall misclassification probability can reach to 74%.


4> Using adversarial training as a defense method to improve robustness of learning models ployed in power systems:

Adversarial training based on FGSM: pqd_adv_training_FGSM.py

Adversarial training based on SSA: pqd_adv_training_SSA.py

robustness_original = np.load("Original_robustness.npy")

robustness_epoch_SSA = np.load("Perturbation_percent_SSA_SSA_mean.npy")

robustness_epoch_FGSM = np.load("Perturbation_percent_FGSM_SSA_mean.npy")

robustness_normal = np.load("Perturbation_percent_normal_mean.npy")

Comparsion of adversarial training:compare_adv_training.py

leraning curve: pqd_learning_curve.py





5> Qualitatively analyze the relationship between the characteristics of adversarial attacks to signals and defense methods 

( Use t-SNE algorithm  to visualize signal measurements and their feature representations learned at different layers of DNNs)

pqd_visualization.py

advtrain_weights_5.h5

adv_SSA_overshoot0.01_all.npy



The datasat is shared at the link below：
https://www.dropbox.com/sh/aprts9x8l2frcjl/AABCuJ3TsJkSSLj2ZixeAyDAa?dl=0


