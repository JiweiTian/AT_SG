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
    print("original test loss and accuracy")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print("done")

    eps = 0.05
    gamma = 0
    with sess.as_default():
    #############################################Adversarial Trainding based on FGSM################################################################################################
        for adv_train_epoch in range(0, 6):

            SCORE0 = list()
            SCORE1 = list()
            SCORE2 = list()

            ###################### Generate Adversarial Signals based on FGSM
            adv_FGSM = []
            Perturbation_percent_FGSM = list()
            Perturbation_percent_FGSM_mean = list()
            counter = 0
            print("FGSM Generating:")
            # Initialize the SGD optimizer
            grad, sign_grad = scaled_gradient(x, y, predictions)
            time_start = time.time()
            length_attack = 25000
            # length_attack = 250
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
            SCORE0.append(score)
            print("test loss and accuracy against adversarial signals based on FGSM")
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            Perturbation_percent_FGSM = np.array(Perturbation_percent_FGSM, dtype=float)
            print("max of Perturbation_percent_FGSM:", Perturbation_percent_FGSM.max())
            print("mean of Perturbation_percent_FGSM:", np.mean(Perturbation_percent_FGSM))
            Perturbation_percent_FGSM_mean.append(np.mean(Perturbation_percent_FGSM))

            teY_pred = np.argmax(model.predict(teX[:length_attack], batch_size=32), axis=1)
            adv_pred = np.argmax(model.predict(adv_FGSM, batch_size=32), axis=1)

            ################################## Adversarial Training Begins
            model.fit(adv_FGSM, teY[:length_attack], batch_size=batch_size, epochs=1, validation_split=0.1, shuffle=True)
            model.save('dnn_clean_FGSM_AT_{}.h5'.format(adv_train_epoch))
            model.save_weights('dnn_clean_FGSM_AT_{}.h5'.format(adv_train_epoch))
            score = model.evaluate(adv_FGSM, teY[:length_attack], verbose=0)
            SCORE1.append(score)
            print("test loss and accuracy of new trainded model against adversarial signals based on FGSM")
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            print('Done')

            print("test loss and accuracy of new trainded model against normal signals")
            score2 = model.evaluate(teX, teY, verbose=0)
            SCORE2.append(score2)
            print('Test loss:', score2[0])
            print('Test accuracy:', score2[1])
            print('Done')


            ################Evaluate the Robustness of the new model based on SSA
            overshoot = 0.01
            Perturbation_percent_SSA = list()
            Perturbation_percent_SSA_mean = list()
            Perturbation_norm = list()
            adv_SSA = []
            max_iter = 50
            nb_candidate = 17
            grad_ssa = ssa_gradient(x, y, predictions)
            counter = 0
            print("SSA Generating:")

            for q in range(length_attack):
                if counter % 50 == 0 and counter > 0:
                    print("Attack on samples" + str(counter))
                TEMP_FIXED = np.copy(teX[counter])
                TEMP_CHANGE = np.copy(teX[counter])

                # Initialize the loop variables
                iteration = 0
                current = teY[counter]
                if current.shape == ():
                    current = np.array([current])
                w = np.squeeze(np.zeros(TEMP_CHANGE.shape[1:]))  # same shape as original image
                r_tot = np.zeros(TEMP_CHANGE.shape)
                current = np.argmax(current)  # class
                original = current  # use original label as the reference

                # Repeat this main loop until we have achieved misclassification
                while (np.any(current == original) and iteration < max_iter):
                    gradients = sess.run(grad_ssa,feed_dict={x: TEMP_CHANGE.reshape(-1, 640,1), y: teY[counter].reshape(-1, 17),keras.backend.learning_phase(): 0})  ### calculate grads
                    predictions_val = sess.run(predictions, feed_dict={x: TEMP_CHANGE.reshape(-1,640,1), y: teY[counter].reshape(-1, 17), keras.backend.learning_phase(): 0})
                    pert = np.inf
                    if np.all(current == original):
                        for k in range(0, nb_candidate):
                            while k != original:
                                w_k = gradients[0, k, ...] - gradients[0, original, ...]
                                f_k = predictions_val[0, k] - predictions_val[0, original]
                                # adding value 0.00001 to prevent f_k = 0
                                #pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten())
                                pert_k = (abs(f_k) + 0.00001) / (np.linalg.norm(w_k.flatten())+ 0.00001)
                                if pert_k < pert:
                                    pert = pert_k
                                    w = w_k
                                break
                        r_i = pert * w / (np.linalg.norm(w) + 0.00001)
                        r_tot[...] = r_tot[...] + r_i

                    #adv_x = np.clip(r_tot + r, clip_min, clip_max)   #### original
                    adv_x = r_tot + TEMP_FIXED     #### my
                    adv_x_pred = np.argmax(model.predict(adv_x.reshape(1, 640,1)), axis=1)
                    current = adv_x_pred
                    TEMP_CHANGE = adv_x
                    # Update loop variables
                    iteration = iteration + 1

                # need to clip this image into the given range
                # adv_x = np.clip((1 + overshoot) * r_tot + r, clip_min, clip_max) #### original
                adv_x = r_tot * (1 + overshoot) + TEMP_FIXED    #### my
                Perturbation_norm.append(np.linalg.norm(adv_x - TEMP_FIXED, keepdims = True))
                perturbation_percent = np.linalg.norm(adv_x - TEMP_FIXED, keepdims = True) / np.linalg.norm(TEMP_FIXED)
                Perturbation_percent_SSA.append(perturbation_percent)
                adv_SSA.append(adv_x)
                counter += 1

            time_end = time.time()
            time_cost = time_end - time_start
            print("time", time_cost)
            adv_SSA = np.array(adv_SSA, dtype=float).reshape(-1, 640)
            adv_SSA = np.expand_dims(adv_SSA, axis=2)
            Perturbation_percent_SSA = np.array(Perturbation_percent_SSA, dtype=float)
            print("max of Perturbation_percent_ssa", Perturbation_percent_SSA.max())
            print("mean of Perturbation_percent_ssa", np.mean(Perturbation_percent_SSA))
            Perturbation_percent_SSA_mean.append(np.mean(Perturbation_percent_SSA))


            adv_train_epoch = adv_train_epoch + 1

        np.save("SCORE0.npy", np.array(SCORE0, dtype=float))
        np.save("SCORE1.npy", np.array(SCORE1, dtype=float))
        np.save("SCORE2.npy", np.array(SCORE2, dtype=float))
        np.save("Perturbation_percent_FGSM_SSA_mean.npy",np.array(Perturbation_percent_SSA_mean, dtype=float))

        print('Done')
    #########################################################################FGSM################################################################################################
