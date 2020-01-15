import numpy as np
import keras

def signal_specific(TEMP_CHANGE,sample_y, x,y,current, original, model, max_iter, grad_deepfool, nb_candidate, overshoot, TEMP_FIXED, predictions,sess, iteration):
    r_tot = np.zeros(TEMP_CHANGE.shape)
    while (np.any(current == original) and iteration < max_iter):
        gradients = sess.run(grad_deepfool,
                             feed_dict={x: TEMP_CHANGE.reshape(-1, 640, 1), y: sample_y.reshape(-1, 17),
                                        keras.backend.learning_phase(): 0})  ### calculate grads
        predictions_val = sess.run(predictions,
                                   feed_dict={x: TEMP_CHANGE.reshape(-1, 640, 1), y: sample_y.reshape(-1, 17),
                                              keras.backend.learning_phase(): 0})
        pert = np.inf
        if np.all(current == original):
            for k in range(0, nb_candidate):
                while k != original:
                    w_k = gradients[0, k, ...] - gradients[0, original, ...]
                    f_k = predictions_val[0, k] - predictions_val[0, original]
                    # adding value 0.00001 to prevent f_k = 0
                    # pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten())
                    pert_k = (abs(f_k) + 0.00001) / (np.linalg.norm(w_k.flatten()) + 0.00001)
                    if pert_k < pert:
                        pert = pert_k
                        w = w_k
                    break
            # r_i = pert * w / np.linalg.norm(w)
            r_i = pert * w / (np.linalg.norm(w) + 0.00001)
            r_tot[...] = r_tot[...] + r_i

        # adv_x = np.clip(r_tot + r, clip_min, clip_max)   #### original
        adv_x = r_tot + TEMP_FIXED  #### my
        adv_x_pred = np.argmax(model.predict(adv_x.reshape(1, 640, 1)), axis=1)
        current = adv_x_pred
        TEMP_CHANGE = adv_x
        # Update loop variables
        iteration = iteration + 1

    # need to clip this image into the given range
    # adv_x = np.clip((1 + overshoot) * r_tot + r, clip_min, clip_max) #### original
    adv_x = r_tot * (1 + overshoot) + TEMP_FIXED  #### my
    return adv_x