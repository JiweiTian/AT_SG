# coding=utf-8
import scipy.io as sio
import numpy as np
import keras
from numpy import shape
import tensorflow as tf
from Neural_Net_Module import dnn_model
from keras.optimizers import SGD
import h5py

import matplotlib.pyplot as plt
import time
from keras.models import load_model
import mpl_toolkits.axisartist as axisartist
from collections import Counter
from sklearn.metrics import confusion_matrix

batch_size=128
nb_epoch=10
eps=0.5   ### eps = 0.5(FGSM) can misclassify all PQ signals
gamma=0

def scaled_gradient(x, y, predictions):
    #loss: the mean of loss(cross entropy)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
    grad, = tf.gradients(loss, x)
    signed_grad = tf.sign(grad)
    return grad, signed_grad


def jacobian_graph(predictions, x, nb_classes):
  """
  Create the Jacobian graph to be ran later in a TF session
  :param predictions: the model's symbolic output (linear output,
      pre-softmax)
  :param x: the input placeholder
  :param nb_classes: the number of classes the model has
  :return:
  """

  # This function will return a list of TF gradients
  list_derivatives = []

  # Define the TF graph elements to compute our derivatives for each class
  for class_ind in range(nb_classes):
    derivatives, = tf.gradients(predictions[:, class_ind], x)
    list_derivatives.append(derivatives)

  return list_derivatives


def ssa_gradient(x, y, predictions):
    # loss: the mean of loss(cross entropy)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
    # grad, = tf.gradients(loss, x)
    grad = tf.stack(jacobian_graph(predictions, x, 17), axis=1)
    return grad


def projection(values, eps, norm_p):
    """
    Project `values` on the L_p norm ball of size `eps`.

    :param values: Array of perturbations to clip.
    :type values: `np.ndarray`
    :param eps: Maximum norm allowed.
    :type eps: `float`
    :param norm_p: L_p norm to use for clipping. Only 1, 2 and `np.Inf` supported for now.
    :type norm_p: `int`
    :return: Values of `values` after projection.
    :rtype: `np.ndarray`
    """
    # Pick a small scalar to avoid division by 0
    tol = 10e-8
    values_tmp = values.reshape((values.shape[0], -1))

    if norm_p == 2:
        values_tmp = values_tmp * np.expand_dims(np.minimum(1., eps / (np.linalg.norm(values_tmp, axis=1) + tol)),
                                                 axis=1)
    elif norm_p == 1:
        values_tmp = values_tmp * np.expand_dims(
            np.minimum(1., eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)), axis=1)
    elif norm_p == np.inf:
        values_tmp = np.sign(values_tmp) * np.minimum(abs(values_tmp), eps)
    else:
        raise NotImplementedError('Values of `norm_p` different from 1, 2 and `np.inf` are currently not supported.')

    values = values_tmp.reshape(values.shape)
    return values

if __name__ == '__main__':


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)


    data = h5py.File('17signal15000.mat','r')
    signals = data['signal'][:]
    signals = np.transpose(signals)

    signals_norm = np.linalg.norm(signals, axis=1, keepdims= True)
    signals_norm_mean = np.mean(signals_norm)
    print("mean norm of signals：", signals_norm_mean)


    labels = np.zeros((15000,1))
    for i in range(1,17):
        label = i * np.ones((15000, 1))
        labels = np.concatenate((labels, label))

    pqd_type = ['Normal','Sag','Swell','Interruption','Transient/Impulse/Spike','Oscillatory transient','Harmonics','Harmonics with Sag','Harmonics with Swell','Flicker','Flicker with Sag','Flicker with Swell','Sag with Oscillatory transient','Swell with Oscillatory transient','Sag with Harmonics','Swell with Harmonics','Notch']

    np.random.seed(10)
    index = np.arange(len(labels))
    np.random.shuffle(index)

    labels = labels[index]
    teY_original = labels[230000:]
    labels = keras.utils.to_categorical(labels, num_classes=None)
    signals = signals[index]
    signals = np.expand_dims(signals, axis=2)

    trX = signals[:230000]
    trY = labels[:230000]
    teX = signals[230000:]
    teY = labels[230000:]

    print("Input label shape", shape(labels))
    print("Input data shape", shape(signals))




    model = dnn_model(input_dim=640)

    x = tf.placeholder(tf.float32, shape=(None, 640, 1))
    y = tf.placeholder(tf.float32, shape=(None, 17))

    #################################logits########################################
    predictions = model(x) ###### after softmax
    predictions_logits = predictions.op.inputs[0]  ###logits, before softmax
    predictions = predictions_logits
    #################################logits#######################################

    #sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
                  metrics=['accuracy'])

    #model.fit(trX, trY, batch_size=batch_size, epochs=nb_epoch,validation_split=0.1, shuffle=True)  # validation_split=0.1
    #model.save_weights('17oldweights_dnn_clean10.h5')
    model.load_weights('17oldweights_dnn_clean10.h5')
    score = model.evaluate(teX, teY, verbose=0)
    print("original test loss and accuracy")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    with sess.as_default():

    #####################################################################SSA################################################################################################
        adv_SSA = np.load("adv_SSA_overshoot0.01_all.npy")
        adv_trX = adv_SSA[:230000]
        adv_teX = adv_SSA[230000:]
        SCORE1 = list()
        SCORE2 = list()
        SCORE3 = list()
        SCORE4 = list()
        # SCORE1 = np.load("22020SCORE1.npy")
        # SCORE2 = np.load("22020SCORE2.npy")
        # SCORE3 = np.load("22020SCORE3.npy")
        # SCORE4 = np.load("22020SCORE4.npy")
        # real = np.array([SCORE1[:,1], SCORE2[:,1]]).T
        # att = np.array([SCORE3[:,1], SCORE4[:,1]]).T
        # ALL = np.concatenate((real, att),axis=1)

        for adv_train_epoch in range(0,10):
            # score = model.evaluate(adv_SSA, labels, verbose=0)
            time_start = time.time()
            score = model.evaluate(trX, trY, verbose=0)
            SCORE1.append(score)
            score = model.evaluate(teX, teY, verbose=0)
            SCORE2.append(score)
            score = model.evaluate(adv_trX, trY, verbose=0)
            SCORE3.append(score)
            score = model.evaluate(adv_teX, teY, verbose=0)
            SCORE4.append(score)
            ############Adversarial Training Begins
            model.fit(adv_trX, trY, batch_size=batch_size, epochs=1, validation_split=0.1, shuffle=True, verbose=2)
            model.save_weights('AT_{}.h5'.format(adv_train_epoch))
            time_end = time.time()
            time_cost = time_end - time_start
            print("time", time_cost)

            adv_train_epoch = adv_train_epoch + 1
        SCORE1 = np.array(SCORE1, dtype=float)
        SCORE2 = np.array(SCORE2, dtype=float)
        SCORE3 = np.array(SCORE3, dtype=float)
        SCORE4 = np.array(SCORE4, dtype=float)
        np.save("2020SCORE1.npy",np.array(SCORE1, dtype=float))
        np.save("2020SCORE2.npy", np.array(SCORE2, dtype=float))
        np.save("2020SCORE3.npy", np.array(SCORE3, dtype=float))
        np.save("2020SCORE4.npy", np.array(SCORE4, dtype=float))
        print('Done')

    #####################################################################SSA################################################################################################
