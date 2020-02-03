# coding=utf-8

import numpy as np
import keras
from numpy import shape
import tensorflow as tf
from Neural_Net_Module import dnn_model
import h5py
import time
from collections import Counter
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import seaborn as sns
from sklearn.metrics import confusion_matrix


batch_size=32
nb_epoch=10



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
        ##########################################################Signal Specific Adversarial Signals Generating####################################################
        # overshoot = 0.01
        # Perturbation_percent_SSA = list()
        # Perturbation_norm = list()
        # SCORE0 = list()
        # adv_SSA = []
        # counter = 0
        # max_iter = 50
        # nb_candidate = 17
        # print("Signal Specific Adversarial Signals Generating:")
        # grad_ssa = ssa_gradient(x, y, predictions)
        # time_start = time.time()
        # length_attack = 25000
        # for q in range(length_attack):
        #     if counter % 50 == 0 and counter > 0:
        #         print("Attack on samples" + str(counter))
        #     TEMP_FIXED = np.copy(teX[counter])
        #     TEMP_CHANGE = np.copy(teX[counter])
        #
        #     # Initialize the loop variables
        #     iteration = 0
        #     current = teY[counter]
        #     if current.shape == ():
        #         current = np.array([current])
        #     w = np.squeeze(np.zeros(TEMP_CHANGE.shape[1:]))  # same shape as original image
        #     r_tot = np.zeros(TEMP_CHANGE.shape)
        #     current = np.argmax(current)  # class
        #     original = current  # use original label as the reference
        #
        #     # Repeat this main loop until we have achieved misclassification
        #     while (np.any(current == original) and iteration < max_iter):
        #         gradients = sess.run(grad_ssa,feed_dict={x: TEMP_CHANGE.reshape(-1, 640,1), y: teY[counter].reshape(-1, 17),keras.backend.learning_phase(): 0})  ### calculate grads
        #         predictions_val = sess.run(predictions, feed_dict={x: TEMP_CHANGE.reshape(-1,640,1), y: teY[counter].reshape(-1, 17), keras.backend.learning_phase(): 0})
        #         pert = np.inf
        #         if np.all(current == original):
        #             for k in range(0, nb_candidate):
        #                 while k != original:
        #                     w_k = gradients[0, k, ...] - gradients[0, original, ...]
        #                     f_k = predictions_val[0, k] - predictions_val[0, original]
        #                     # adding value 0.00001 to prevent f_k = 0
        #                     #pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten())
        #                     pert_k = (abs(f_k) + 0.00001) / (np.linalg.norm(w_k.flatten())+ 0.00001)
        #                     if pert_k < pert:
        #                         pert = pert_k
        #                         w = w_k
        #                     break
        #             #r_i = pert * w / np.linalg.norm(w)
        #             r_i = pert * w / (np.linalg.norm(w) + 0.00001)
        #             r_tot[...] = r_tot[...] + r_i
        #
        #         #adv_x = np.clip(r_tot + r, clip_min, clip_max)   #### original
        #         adv_x = r_tot + TEMP_FIXED     #### my
        #         adv_x_pred = np.argmax(model.predict(adv_x.reshape(1, 640,1)), axis=1)
        #         current = adv_x_pred
        #         TEMP_CHANGE = adv_x
        #         # Update loop variables
        #         iteration = iteration + 1
        #
        #     # need to clip this image into the given range
        #     # adv_x = np.clip((1 + overshoot) * r_tot + r, clip_min, clip_max) #### original
        #     adv_x = r_tot * (1 + overshoot) + TEMP_FIXED    #### my
        #     Perturbation_norm.append(np.linalg.norm(adv_x - TEMP_FIXED, keepdims = True))
        #     perturbation_percent = np.linalg.norm(adv_x - TEMP_FIXED, keepdims = True) / np.linalg.norm(TEMP_FIXED)
        #     Perturbation_percent_SSA.append(perturbation_percent)
        #     adv_SSA.append(adv_x)
        #     counter += 1
        #
        # time_end = time.time()
        # time_cost = time_end - time_start
        # print("time", time_cost)
        # adv_SSA = np.array(adv_SSA, dtype=float).reshape(-1, 640)
        # adv_SSA = np.expand_dims(adv_SSA, axis=2)
        # Perturbation_percent_SSA = np.array(Perturbation_percent_SSA, dtype=float)
        # print("max of Perturbation_percent_ssa", Perturbation_percent_SSA.max())
        # print("mean of Perturbation_percent_ssa", np.mean(Perturbation_percent_SSA))
        #
        # score = model.evaluate(adv_SSA, teY[:length_attack], verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])
        # print('Done')
        # np.save("adv_SSA.npy", adv_SSA)
        ##########################################################Signal Specific Adversarial Signals Generating####################################################

        #######confusion matrxi of SSA##################

        pqd_predict_normal = np.argmax(model.predict(teX), axis=1)
        adv_univer = np.load("adv_SSA.npy")
        pqd_predict_attack = np.argmax(model.predict(adv_univer), axis=1)
        sns.set()
        f, ax = plt.subplots()
        C2 = confusion_matrix(teY_original, pqd_predict_attack, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])


        C2 = C2 / C2.sum(axis=1).reshape(17,1)
        np.save("confusion_matrix_SSA.npy", np.array(C2, dtype=float))
        # C2 = np.load("confusion_matrix.npy")
        sns.heatmap(C2, annot = True, ax=ax,linewidths='0.5', cmap="BuPu",
                    xticklabels =['C-1','C-2','C-3','C-4','C-5','C-6','C-7','C-8','C-9','C-10','C-11','C-12','C-13','C-14','C-15','C-16','C-17'],
                    yticklabels =['C-1','C-2','C-3','C-4','C-5','C-6','C-7','C-8','C-9','C-10','C-11','C-12','C-13','C-14','C-15','C-16','C-17'])  # 画热力图
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 30,
                 }
        ax.set_title('Confusion matrix',font2)
        ax.set_xlabel('Predict',font2)
        ax.set_ylabel('True',font2)
        plt.show()
        #######confusion matrxi of SSA##################




        ##########################################################Samples Generated by Signal Specific Attack#######################################################

        adv_SSA = np.load("adv_SSA.npy")
        length_attack = 25000

        Pert_ratio = np.linalg.norm(np.squeeze(adv_SSA - teX),axis=1,keepdims=True)/np.linalg.norm(np.squeeze(teX),axis=1,keepdims=True)
        Pert_ratio_mean = np.mean(Pert_ratio)

        pqd_predict_normal = np.argmax(model.predict(teX), axis=1)
        pqd_predict_attack_df = np.argmax(model.predict(adv_SSA), axis=1)

        index = np.arange(0, 25000)
        same_index = index[pqd_predict_normal == pqd_predict_attack_df]
        same = [pqd_predict_normal[same_index,], pqd_predict_attack_df[same_index,]]
        same = np.transpose(np.array(same))

        result_same_df = [Counter(same[:, i]).most_common() for i in range(same.shape[1])]


        adv_SSA = np.load("adv_SSA.npy")
        pqd_17_index = [35, 13, 14, 31, 67, 18, 30, 38, 19, 17, 2, 3, 15, 26, 37, 5, 12]  ###from clss 0-16（1-17）
        pqd_17_predict_normal = np.argmax(model.predict(teX[pqd_17_index,]), axis=1) ### the original prediction before attack
        pqd_17_predict_attack = np.argmax(model.predict(adv_SSA[pqd_17_index,]), axis=1) ### the new prediction after attack
        fig = plt.figure()
        for i in range(17):
            plt.subplot(3, 6, i+1)
            plt.plot(teX[pqd_17_index[i],], 'b--',label = 'Original')
            plt.plot(adv_SSA[pqd_17_index[i],], 'r:',label = 'SSA')
            plt.legend()
            plt.title("C-{},{}\n(C-{},{})".format(pqd_17_predict_attack[i]+1, pqd_type[pqd_17_predict_attack[i]], pqd_17_predict_normal[i]+1, pqd_type[pqd_17_predict_normal[i]]),fontsize=10)
            #plt.title('{}'.format(pqd_type[i]))
        plt.subplots_adjust(wspace =0.2, hspace = 0.3)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
        ax = axisartist.Subplot(fig, 3, 6, 18)
        fig.add_axes(ax)
        ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
        ax.axis["left"].set_axisline_style("-|>", size=1.5)
        ax.axis["top"].set_visible(False)
        ax.axis["bottom"].toggle(all=False, label=True)
        ax.axis["bottom"].label.set_text("Time (s)")
        ax.axis["right"].set_visible(False)
        ax.axis["left"].toggle(all=False, label=True)
        ax.axis["left"].label.set_text("Voltage/Current (V/A p.u.)")
        plt.show()

        print('Done')


