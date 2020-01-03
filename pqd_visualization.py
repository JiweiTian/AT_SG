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
import mpl_toolkits.axisartist as axisartist
from collections import Counter
import seaborn as sns
from sklearn.metrics import confusion_matrix

from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import manifold, datasets, decomposition, ensemble, random_projection
from time import time

from keras.models import Model
from scipy.spatial.distance import cdist

batch_size=32
nb_epoch=10
eps=0.5
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
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')


    #sess = tf.Session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


    keras.backend.set_session(sess)

    # data = sio.loadmat(r'signal.mat')  # 把这个路径改成你的mat路径即可
    # signals = data['signal']

    data = h5py.File('17signal15000.mat','r')
    signals = data['signal'][:]
    signals = np.transpose(signals)

    signals_norm = np.linalg.norm(signals, axis=1, keepdims= True)
    signals_norm_mean = np.mean(signals_norm)
    print("mean of norm of signals：", signals_norm_mean)

    # data = tables.openFile('signal.mat')

    # mat = matlab.engine.start_matlab()
    # data = mat.load("dataset.mat", nargout=1)




    labels = np.zeros((15000,1))
    for i in range(1,17):
        label = i * np.ones((15000, 1))
        labels = np.concatenate((labels, label))

    pqd_type = ['Normal','Sag','Swell','Interruption','Transient/Impulse/Spike','Oscillatory transient','Harmonics','Harmonics with Sag','Harmonics with Swell','Flicker','Flicker with Sag','Flicker with Swell','Sag with Oscillatory transient','Swell with Oscillatory transient','Sag with Harmonics','Swell with Harmonics','Notch']



    np.random.seed(10)

    index = np.arange(len(labels))
    np.random.shuffle(index)

    labels = labels[index]
    original_labels = labels

    teY_original = labels[230000:]
    labels = keras.utils.to_categorical(labels, num_classes=None)

    signals = signals[index]
    original_signals = signals
    signals = np.expand_dims(signals, axis=2)
    # labels = np.expand_dims(labels, axis=2)

    def plot_embedding_2d(X, y_data, title=None, number=None, colorbar=None):
        x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
        X = (X - x_min) / (x_max - x_min)

        vis_x = X[:, 0]
        vis_y = X[:, 1]

        plt.scatter(vis_x, vis_y, c=y_data, s=1, cmap=plt.cm.get_cmap("gist_rainbow", 17))
        if number == 1:
            plt.xlabel("(a)", fontsize=25)
        elif number == 2:
            plt.xlabel("(b)", fontsize=25)
        else:
            plt.xlabel("(c)", fontsize=25)

        if colorbar == 1:
            plt.colorbar(ticks=range(17))

        plt.clim(-0.5, 16.5)



    def plot_embedding_3d(X, y_data, title=None):
        x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
        X = (X - x_min) / (x_max - x_min)
        fig = plt.figure()
        ax = Axes3D(fig)
        vis_x = X[:, 0]
        vis_y = X[:, 1]
        vis_z = X[:, 2]

        ax.scatter(vis_x, vis_y, vis_z, c=y_data, s=1, cmap=plt.cm.get_cmap("gist_rainbow", 17))

        if title is not None:
            plt.title(title)


    def NeighborhoodHit(X_embedded, C, n_neighbors=5):

        dist_X_embedded = cdist(X_embedded, X_embedded, p=2.)

        # select the kNN for each instance
        ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1]
        m = X_embedded.shape[0]

        def ratio(x, kNN):  # indices
            # if the class of the KNN belongs to the class of the point at evaluation
            same_class = len(np.where(C[kNN] == C[x])[0])
            return same_class

        NH = 0.0
        for x in range(m):  # for all the examples
            NH += ratio(x, ind_X_embedded[x])
        NH = NH / (float(m) * float(n_neighbors))
        return NH



    print("Input label shape", shape(labels))
    print("Input data shape", shape(signals))

    trX = signals[:230000]
    trY = labels[:230000]
    teX = signals[230000:]
    teY = labels[230000:]



    model = dnn_model(input_dim=640)


    ##### the last hidden layer

    visualize_layer_model = Model(inputs=model.input, outputs=model.get_layer('visualize').output)



    x = tf.placeholder(tf.float32, shape=(None, 640, 1))
    y = tf.placeholder(tf.float32, shape=(None, 17))

    #################################有无logits########################################
    predictions = model(x) ###### 非logits, softmax之后的数据
    predictions_logits = predictions.op.inputs[0]  ###logits, softmax之前的数据
    predictions = predictions_logits
    #################################有无logits#######################################

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
                  metrics=['accuracy'])

    print("random cnn model:")
    score = model.evaluate(teX, teY, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    number_visualize = 1000

    y_data = original_labels[0:number_visualize]
    y_data = y_data.astype(np.int32).reshape(number_visualize, )

    ###########################################################Figure 1#################################################

    fig1 = plt.figure()

    plt.subplot(1, 3, 1)
    print("normal original data:")
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    print(X_tsne.shape)
    plot_embedding_2d(X_tsne[:, 0:2], y_data, "t-SNE 2D", 1, 0)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    print("NH of normal original data:", NH)
    plt.xticks([])
    plt.yticks([])


    plt.subplot(1, 3, 2)
    print("normal hidden output data (random cnn model):")
    original_signals = visualize_layer_model.predict(signals)
    # np.save("visualize_radmom_cnn_signals_output.npy", np.array(original_signals, dtype=float))
    # original_signals = np.load("visualize_radmom_cnn_signals_output.npy")
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    print(X_tsne.shape)
    plot_embedding_2d(X_tsne[:, 0:2], y_data, "t-SNE 2D", 2, 0)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    print("NH of normal hidden output data (random cnn model):", NH)
    plt.xticks([])
    plt.yticks([])

    print("trained cnn model:")
    model.load_weights('17oldweights_dnn_clean10.h5')
    score = model.evaluate(teX, teY, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plt.subplot(1, 3, 3)
    print("normal hidden ouotput data:")
    original_signals = visualize_layer_model.predict(signals)
    # original_signals = np.load("visualize_output.npy")
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    print(X_tsne.shape)
    plot_embedding_2d(X_tsne[:, 0:2], y_data, "t-SNE 2D", 3, 1)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    print("NH of normal hidden data:", NH)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.show()
    print("Figure 1 done")
    ###########################################################Figure 1#################################################

    ###########################################################Figure 2#################################################
    '''fig2 = plt.figure()

    plt.subplot(1, 3, 1)
    print("SSA original data:")
    original_signals = np.load("./adv_SSA_overshoot0.01_all.npy")
    original_signals = np.squeeze(original_signals)
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    plot_embedding_2d(X_tsne[:, 0:2], y_data, "t-SNE 2D", 1, 0)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    print("NH of SSA original data:", NH)
    plt.xticks([])
    plt.yticks([])

    print("trained cnn model:")
    model.load_weights('17oldweights_dnn_clean10.h5')
    score = model.evaluate(teX, teY, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    plt.subplot(1, 3, 2)
    print("SSA hidden ouotput data:")
    adv_SSA_signals = np.load("./adv_SSA_overshoot0.01_all.npy")
    visualize_adv_SSA_signals_output = visualize_layer_model.predict(adv_SSA_signals)
    # np.save("visualize_adv_SSA_signals_output.npy", np.array(visualize_adv_SSA_signals_output, dtype=float))
    # original_signals = np.load("visualize_adv_SSA_signals_output.npy")
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    plot_embedding_2d(X_tsne[:, 0:2], y_data, "t-SNE 2D", 2, 0)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    print("NH of SSA hidden output data:", NH)
    plt.xticks([])
    plt.yticks([])

    print("adv trained cnn model:")
    model.load_weights('advtrain_weights_5.h5')
    score = model.evaluate(teX, teY, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plt.subplot(1, 3, 3)
    print("SSA hidden ouotput data (adv trained cnn):")
    adv_SSA = np.load("./adv_SSA_overshoot0.01_all.npy")
    score = model.evaluate(adv_SSA, labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    original_signals = visualize_layer_model.predict(adv_SSA)
    # np.save("visualize_advtrain_cnn_ssa_signals_output.npy", np.array(original_signals, dtype=float))
    # original_signals = np.load("visualize_advtrain_cnn_ssa_signals_output.npy")
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    plot_embedding_2d(X_tsne[:, 0:2], y_data, "t-SNE 2D", 3, 1)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    print("NH of SSA hidden output data:", NH)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()'''

    ###########################################################Figure 2#################################################

    ###########################################################Figure 3#################################################
    '''fig3 = plt.figure()

    plt.subplot(1, 3, 1)
    print("universal input data :")
    universal_pert = np.load("universal_pert_5000_1_2.0.npy")
    adv_univer = signals + universal_pert
    original_signals = np.squeeze(adv_univer)
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    plot_embedding_2d(X_tsne[:, 0:2], y_data, "t-SNE 2D", 1, 0)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    print("NH of SSA hidden output data:", NH)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 2)
    print("universal hidden output data :")
    universal_pert = np.load("universal_pert_5000_1_2.0.npy")
    adv_univer = signals + universal_pert
    original_signals = visualize_layer_model.predict(adv_univer)
    # np.save("visualize_universal_cnn_signals_output.npy", np.array(original_signals, dtype=float))
    # original_signals = np.load("visualize_universal_cnn_signals_output.npy")
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    plot_embedding_2d(X_tsne[:, 0:2], y_data, "t-SNE 2D", 2, 0)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    print("NH of SSA hidden output data:", NH)
    plt.xticks([])
    plt.yticks([])

    print("adv trained cnn model:")
    model.load_weights('advtrain_weights_5.h5')
    score = model.evaluate(teX, teY, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plt.subplot(1, 3, 3)
    print("universal hidden ouotput data (adv trained cnn):")
    universal_pert = np.load("universal_pert_5000_1_2.0.npy")
    adv_univer = signals + universal_pert
    score = model.evaluate(adv_univer, labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    original_signals = visualize_layer_model.predict(adv_univer)
    # np.save("visualize_universal_advcnn_signals_output.npy", np.array(original_signals, dtype=float))
    # original_signals = np.load("visualize_universal_advcnn_signals_output.npy")
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    plot_embedding_2d(X_tsne[:, 0:2], y_data, "t-SNE 2D", 3, 1)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    print("NH of SSA hidden output data:", NH)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()'''
    ###########################################################Figure 3#################################################


    print("done")