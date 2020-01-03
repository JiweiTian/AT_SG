# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt


robustness_original = np.load("Original_robustness.npy")

robustness_epoch_SSA = np.load("Perturbation_percent_SSA_SSA_mean.npy")
robustness_epoch_SSA = np.hstack((robustness_original,robustness_epoch_SSA))

robustness_epoch_FGSM = np.load("Perturbation_percent_FGSM_SSA_mean.npy")
robustness_epoch_FGSM = np.hstack((robustness_original, robustness_epoch_FGSM))

robustness_normal = np.load("Perturbation_percent_normal_mean.npy")
robustness_normal = np.hstack((robustness_original, robustness_normal))

fig = plt.figure()
plt.plot(robustness_epoch_SSA, 'b-', label='SSA')
plt.plot(robustness_epoch_FGSM, 'g-', label='FGSM')
plt.plot(robustness_normal, 'r--', label='Clean')
plt.xlabel('Number of extra epochs',fontsize=15)
plt.ylabel(r'$\hat{\rho}_{adv}$',fontsize=15)
plt.xlim(0, 5)
plt.legend(prop={'size':15})
plt.show()


print("done")