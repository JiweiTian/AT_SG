# coding=utf-8

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

batch_size=32
nb_epoch=10
eps=0.5   ###0.4FGSM不能都成功，0.5FGSM能都成功(17种类型都改变分类结果)
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


def deepfool_gradient(x, y, predictions):
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
    print("mean of norm of signals：", signals_norm_mean)




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
    # labels = np.expand_dims(labels, axis=2)

    print("Input label shape", shape(labels))
    print("Input data shape", shape(signals))

    trX = signals[:230000]
    trY = labels[:230000]
    teX = signals[230000:]
    teY = labels[230000:]



    model = dnn_model(input_dim=640)

    x = tf.placeholder(tf.float32, shape=(None, 640, 1))
    y = tf.placeholder(tf.float32, shape=(None, 17))

    #################################logits########################################
    predictions = model(x)
    predictions_logits = predictions.op.inputs[0]
    predictions = predictions_logits
    #################################有无logits#######################################

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
                  metrics=['accuracy'])

    #model.fit(trX, trY, batch_size=batch_size, epochs=nb_epoch,validation_split=0.1, shuffle=True)  # validation_split=0.1
    #model.save('17oldmodel_dnn_clean10.h5')
    #model = load_model('17model_dnn_clean5.h5')
    #model.save_weights('17oldweights_dnn_clean10.h5')
    model.load_weights('17oldweights_dnn_clean10.h5')
    score = model.evaluate(teX, teY, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    print("done")


    with sess.as_default():
        score = model.evaluate(teX, teY, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        score1 = np.load("score1.npy")
        score2 = np.load("score2.npy")
        score3 = np.load("score3.npy")
        score4 = np.load("score4.npy")
        #########################################plot learning cureve###################################################
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(score1[0:11, 1], 'r', label='Original (train)')
        ax1.plot(score2[0:11, 1], 'r--', label='Original (test)')
        ax1.plot(score3[0:11, 1], 'b', label='SSA (train)')
        ax1.plot(score4[0:11, 1], 'b--', label='SSA (test)')
        ax1.set_xlabel('Epoch',fontsize=15)
        ax1.set_ylabel('Accuracy',fontsize=20)
        ax1.set_xlim(0.0, 10)
        ax1.set_ylim(0.0, 1.0)
        plt.legend(prop={'size':15})

        ax2 = ax1.twinx()
        ax2.plot(score1[0:11, 0], 'm', label='Original (train)')
        ax2.plot(score2[0:11, 0], 'm--', label='Original (test)')
        ax2.plot(score3[0:11, 0], 'g', label='SSA (train)')
        ax2.plot(score4[0:11, 0], 'g--', label='SSA (test)')
        ax2.set_ylim(0.0, 5.0)
        ax2.set_ylabel('Loss', fontsize=20)

        legend1 = ax1.legend(loc=(.10, .6), fontsize=8, shadow=True)
        legend2 = ax2.legend(loc=(.75, .18), fontsize=8, shadow=True)
        legend1.get_frame().set_facecolor('#FFFFFF')
        legend2.get_frame().set_facecolor('#FFFFFF')

        plt.show()
        #########################################plot learning cureve###################################################

        print('Done')