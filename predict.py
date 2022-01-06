# from keras.models import load_model
# from keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import numpy as np
import cv2
import math


def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true)))


def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    return 20 * math.log10(1 / math.sqrt(mse))


if __name__ == '__main__':
    model = load_model('out/model/ddsrcnn_model_30.h5', custom_objects={'PSNRLoss': PSNRLoss})
    X_train = np.zeros((1, 256, 256, 1), dtype=np.uint8)
    low_train = cv2.imread('open1.png', cv2.IMREAD_GRAYSCALE)
    X_train[0] = np.expand_dims(low_train, axis=2)
    preds_test = model.predict(X_train, verbose=1).astype(np.uint8)
    print(preds_test.shape)
    preds_test = preds_test.reshape(256, 256)
    cv2.imshow('s', preds_test)
    cv2.waitKey(0)
    print(psnr(low_train, preds_test))

