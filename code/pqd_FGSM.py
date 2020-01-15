# coding=utf-8
import scipy.io as sio
import numpy as np
import keras
from numpy import shape
import tensorflow as tf
from Neural_Net_Module import dnn_model
from keras.optimizers import SGD
import h5py
import tables
import matlab
import matplotlib.pyplot as plt
import time
from keras.models import load_model
from signal_specific import signal_specific
import mpl_toolkits.axisartist as axisartist
from collections import Counter
import seaborn as sns
from sklearn.metrics import confusion_matrix

batch_size=32
nb_epoch=10
eps=0.5   ### eps = 0.5(FGSM) can misclassify all PQ signals
gamma=0

def scaled_gradient(x, y, predictions):
    #loss: the mean of loss(cross entropy)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
    grad, = tf.gradients(loss, x)
    signed_grad = tf.sign(grad)
    return grad, signed_grad


if __name__ == '__main__':
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')


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
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    print("done")

    with sess.as_default():
    #########################################################FGSM Adversarial Signals Generating####################################################
        adv_FGSM = []
        Perturbation_percent_FGSM = list()
        Perturbation_percent_FGSM_mean = list()
        counter = 0
        print("FGSM Generating:")
        # Initialize the SGD optimizer
        grad, sign_grad = scaled_gradient(x, y, predictions)
        time_start = time.time()
        #length_attack = 25000
        length_attack = 100
        for q in range(length_attack):
            if counter % 50 == 0 and counter > 0:
                print("Attack on samples" + str(counter))
            TEMP_FIXED = np.copy(teX[counter])
            TEMP_CHANGE = np.copy(teX[counter])
            P, P_logits = sess.run([predictions, predictions_logits], feed_dict={x: TEMP_CHANGE.reshape(-1, 640, 1),
                                                                                 keras.backend.learning_phase(): 0})
            gradient_value, signed_grad, P, P_logits = sess.run([grad, sign_grad, predictions, predictions_logits],
                                                                feed_dict={x: TEMP_CHANGE.reshape(-1, 640, 1),
                                                                           y: teY[counter].reshape(-1, 17),
                                                                           keras.backend.learning_phase(): 0})
            saliency_mat = np.abs(gradient_value)
            saliency_mat = (saliency_mat > np.percentile(np.abs(gradient_value), [gamma])).astype(int)
            TEMP_CHANGE = TEMP_CHANGE + np.multiply(eps * signed_grad, saliency_mat)
            adv_FGSM.append(TEMP_CHANGE)
            perturbation_percent = np.linalg.norm(TEMP_CHANGE - TEMP_FIXED, keepdims=True) / np.linalg.norm(TEMP_FIXED)
            Perturbation_percent_FGSM.append(perturbation_percent)

            counter += 1

        time_end = time.time()
        time_cost = time_end - time_start
        print("time", time_cost)

        adv_FGSM = np.array(adv_FGSM, dtype=float).reshape(-1, 640)
        adv_FGSM = np.expand_dims(adv_FGSM, axis=2)
        score = model.evaluate(adv_FGSM, teY[:length_attack], verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        Perturbation_percent_FGSM = np.array(Perturbation_percent_FGSM, dtype=float)
        print("max of Perturbation_percent_FGSM", Perturbation_percent_FGSM.max())
        print("mean of Perturbation_percent_FGSM", np.mean(Perturbation_percent_FGSM))

        teY_pred = np.argmax(model.predict(teX[:length_attack], batch_size=32), axis=1)
        adv_pred = np.argmax(model.predict(adv_FGSM, batch_size=32), axis=1)

        Pert_ratio = np.linalg.norm(np.squeeze(adv_FGSM - teX[:length_attack]), axis=1, keepdims=True) / np.linalg.norm(np.squeeze(teX[:length_attack]),
                                                                                                      axis=1, keepdims=True)
        Pert_ratio_mean = np.mean(Pert_ratio)
    #########################################################FGSM Adversarial Signals Generating####################################################

    ##########################################################Samples Generated by FGSMk############################################################
        pqd_17_index = [35, 13, 14, 31, 67, 18, 30, 38, 19, 17, 2, 3, 15, 26, 37, 5, 12]  ###from clss 0-16（1-17）
        pqd_17_predict_normal = np.argmax(model.predict(teX[pqd_17_index,]), axis=1)  ### the original prediction before attack
        pqd_17_predict_attack_FGSM = np.argmax(model.predict(adv_FGSM[pqd_17_index,]), axis=1) ### the new prediction after attack
        fig = plt.figure()
        for i in range(17):
            plt.subplot(3, 6, i + 1)
            plt.plot(teX[pqd_17_index[i],], 'b--', label='Original')
            plt.plot(adv_FGSM[pqd_17_index[i],], 'r:', label='FGSM')
            plt.legend()
            plt.title("C-{},{}\n(C-{},{})".format(pqd_17_predict_attack_FGSM[i] +1, pqd_type[pqd_17_predict_attack_FGSM[i]], pqd_17_predict_normal[i]+1, pqd_type[pqd_17_predict_normal[i]]),fontsize=10)
        ax = axisartist.Subplot(fig, 3, 6, 18)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
        fig.add_axes(ax)
        ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
        ax.axis["left"].set_axisline_style("-|>", size=1.5)
        ax.axis["top"].set_visible(False)
        ax.axis["bottom"].toggle(all=False, label=True)
        ax.axis["bottom"].label.set_text("Time (s)")
        ax.axis["right"].set_visible(False)
        ax.axis["left"].toggle(all=False, label=True)
        ax.axis["left"].label.set_text("Voltage/Current (V/A p.u.)")
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        plt.show()
        print("done")
    ##########################################################Samples Generated by FGSMk############################################################
