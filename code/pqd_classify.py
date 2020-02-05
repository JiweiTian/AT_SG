# coding=utf-8

import numpy as np
import keras
from numpy import shape
import tensorflow as tf
from Neural_Net_Module import dnn_model
import h5py

batch_size=32
nb_epoch=20

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

    #############################figure:pqd_types##################################
    # fig = plt.figure()
    # for i in range(1,18):
    #     plt.subplot(3, 6, i)
    #     plt.plot(signals[5 + 15000 * (i-1),])
    #     plt.title('{}(C{})'.format (pqd_type[i-1], i),fontsize = 10)
    # ax = axisartist.Subplot(fig, 3, 6, 18)
    # fig.add_axes(ax)
    # ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
    # ax.axis["left"].set_axisline_style("-|>", size=1.5)
    # ax.axis["top"].set_visible(False)
    # ax.axis["bottom"].toggle(all=False, label = True)
    # ax.axis["bottom"].label.set_text("Time (s)")
    # ax.axis["right"].set_visible(False)
    # ax.axis["left"].toggle(all=False, label = True)
    # ax.axis["left"].label.set_text("Voltage/Current (V/A p.u.)")
    #
    # plt.show()
    #############################figure:pqd_types##################################

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

    #sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
                  metrics=['accuracy'])

    # model.fit(trX, trY, batch_size=batch_size, epochs=nb_epoch,validation_split=0.1, shuffle=True)  # validation_split=0.1
    #model.save_weights('17oldweights_dnn_clean10.h5')
    model.load_weights('17oldweights_dnn_clean10.h5')
    score = model.evaluate(teX, teY, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print("done")
