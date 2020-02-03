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

    def plot_embedding_2d(ax, X, y_data, title=None, number=None, colorbar=None, nh = None, accuracy = None):
        x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
        X = (X - x_min) / (x_max - x_min)

        vis_x = X[:, 0]
        vis_y = X[:, 1]

        cax = ax.scatter(vis_x, vis_y, c=y_data, s=1, cmap=plt.cm.get_cmap("gist_rainbow", 17))
        nh = round(nh * 100,2)
        accuracy = round(accuracy * 100,2)
        if number in "adg":
            ax.set_xlabel("({})NH: {}%".format(number, nh), fontsize=17)
        else:
            ax.set_xlabel("({})NH: {}%, Acc: {}%".format(number, nh, accuracy), fontsize=17)

        if colorbar == 1:
            cbar = plt.colorbar(cax, ticks= range(17))
            cbar.mappable.set_clim(-0.5, 16.5)
            cbar.ax.set_yticklabels(['C-1','C-2','C-3','C-4','C-5','C-6','C-7','C-8','C-9','C-10','C-11','C-12','C-13','C-14','C-15','C-16','C-17'])




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
    fig = plt.figure(figsize=(12, 18))

    ax = fig.add_subplot(3, 3, 1)
    print("normal original data:")
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    print(X_tsne.shape)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    plot_embedding_2d(ax, X_tsne[:, 0:2], y_data, "t-SNE 2D", 'a', 0, NH, 0)
    print("NH of normal original data:", NH)
    ax.set_xticks([])
    ax.set_yticks([])


    ax = fig.add_subplot(3, 3, 2)
    print("normal hidden output data (random cnn model):")
    score = model.evaluate(signals[0:number_visualize], labels[0:number_visualize], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    original_signals = visualize_layer_model.predict(signals)
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    print(X_tsne.shape)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    plot_embedding_2d(ax, X_tsne[:, 0:2], y_data, "t-SNE 2D", 'b', 0, NH, score[1])
    print("NH of normal hidden output data (random cnn model):", NH)
    ax.set_xticks([])
    ax.set_yticks([])

    print("trained cnn model:")
    model.load_weights('17oldweights_dnn_clean10.h5')
    ax = fig.add_subplot(3, 3, 3)
    score = model.evaluate(signals[0:number_visualize], labels[0:number_visualize], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("normal hidden output data:")
    original_signals = visualize_layer_model.predict(signals)
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    print(X_tsne.shape)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    plot_embedding_2d(ax, X_tsne[:, 0:2], y_data, "t-SNE 2D", 'c', 1, NH, score[1])
    print("NH of normal hidden data:", NH)
    ax.set_xticks([])
    ax.set_yticks([])


    ax = fig.add_subplot(3, 3, 4)
    print("SSA original data:")
    original_signals = np.load("./adv_SSA_overshoot0.01_all.npy")
    original_signals = np.squeeze(original_signals)
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    plot_embedding_2d(ax, X_tsne[:, 0:2], y_data, "t-SNE 2D", 'd', 0, NH,0)
    print("NH of SSA original data:", NH)
    ax.set_xticks([])
    ax.set_yticks([])

    print("trained cnn model:")
    model.load_weights('17oldweights_dnn_clean10.h5')
    ax = fig.add_subplot(3, 3, 5)
    print("SSA hidden ouotput data:")
    adv_SSA_signals = np.load("./adv_SSA_overshoot0.01_all.npy")
    score = model.evaluate(adv_SSA_signals[0:number_visualize], labels[0:number_visualize], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    original_signals = visualize_layer_model.predict(adv_SSA_signals)
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    plot_embedding_2d(ax, X_tsne[:, 0:2], y_data, "t-SNE 2D", 'e', 0, NH, score[1])
    print("NH of SSA hidden output data:", NH)
    ax.set_xticks([])
    ax.set_yticks([])

    print("adv trained cnn model:")
    model.load_weights('advtrain_weights_5.h5')
    ax = fig.add_subplot(3, 3, 6)
    print("SSA hidden output data (adv trained cnn):")
    adv_SSA = np.load("./adv_SSA_overshoot0.01_all.npy")
    original_signals = visualize_layer_model.predict(adv_SSA)
    score = model.evaluate(adv_SSA[0:number_visualize], labels[0:number_visualize], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    original_signals = visualize_layer_model.predict(adv_SSA_signals)
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    plot_embedding_2d(ax, X_tsne[:, 0:2], y_data, "t-SNE 2D", 'f', 1, NH, score[1])
    print("NH of SSA hidden output data:", NH)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(3, 3, 7)
    print("universal input data :")
    universal_pert = np.load("universal_pert_5000_1_2.0.npy")
    adv_univer = signals + universal_pert
    original_signals = np.squeeze(adv_univer)
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    plot_embedding_2d(ax, X_tsne[:, 0:2], y_data, "t-SNE 2D",'g', 0, NH,0)
    print("NH of SSA hidden output data:", NH)
    ax.set_xticks([])
    ax.set_yticks([])

    print("trained cnn model:")
    model.load_weights('17oldweights_dnn_clean10.h5')
    ax = fig.add_subplot(3, 3, 8)
    print("universal hidden output data :")
    universal_pert = np.load("universal_pert_5000_1_2.0.npy")
    adv_univer = signals + universal_pert
    score = model.evaluate(adv_univer[0:number_visualize], labels[0:number_visualize], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    original_signals = visualize_layer_model.predict(adv_univer)
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    plot_embedding_2d(ax, X_tsne[:, 0:2], y_data, "t-SNE 2D", 'h', 0, NH, score[1])
    print("NH of SSA hidden output data:", NH)
    ax.set_xticks([])
    ax.set_yticks([])

    print("adv trained cnn model:")
    model.load_weights('advtrain_weights_5.h5')
    ax = fig.add_subplot(3, 3, 9)
    print("universal hidden ouotput data (adv trained cnn):")
    universal_pert = np.load("universal_pert_5000_1_2.0.npy")
    adv_univer = signals + universal_pert
    score = model.evaluate(adv_univer[0:number_visualize], labels[0:number_visualize], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    score = model.evaluate(adv_univer, labels, verbose=0)
    original_signals = visualize_layer_model.predict(adv_univer)
    X_data = original_signals[0:number_visualize]
    t0 = time()
    X_tsne = tsne.fit_transform(X_data)
    NH = NeighborhoodHit(X_tsne[:, 0:2], y_data, n_neighbors=5)
    plot_embedding_2d(ax, X_tsne[:, 0:2], y_data, "t-SNE 2D", 'i', 1, NH, score[1])
    print("NH of SSA hidden output data:", NH)
    ax.set_xticks([])
    ax.set_yticks([])


    plt.tight_layout()
    plt.subplots_adjust(left=0.01, top=0.98, bottom=0.05, hspace = 0.15, wspace=0.02)
    plt.show()
    ###########################################################Figure 3#################################################


    print("done")