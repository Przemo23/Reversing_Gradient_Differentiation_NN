import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
import random


def compare_weight_vectors(w1, w2):
    for i in range(len(w1)):
        if (w1[i] == w2[i]).all() is False:
            return False
    return True


def create_Binary_Dataset():
    x = tf.random.normal([512, 2], 2, 1.0, tf.float32, seed=1).numpy()
    x = np.concatenate((x, tf.random.normal([512, 2], -2, 1.0, tf.float32, seed=1).numpy()), axis=0)
    y = np.concatenate((np.ones(512), np.zeros(512)), axis=0)
    x, y = shuffle(x, y, random_state=0)
    # Create batches
    x = np.reshape(x, (32, 32, 2))
    y = np.reshape(y, (32, 32)).astype('int64')
    y = to_categorical(y, num_classes=2)

    return x, y


def create_Reg_Dataset():
    x = (tf.random.uniform([1, 1024], -1.0, 1.0, tf.float32, seed=1) )
    y = np.copy(x) * 0.5 - tf.random.normal([1, 1024], 0.0, 0.05,tf.float32,seed=2).numpy()
    x = np.reshape(x, (32, 32))
    y = np.reshape(y, (32, 32))

    return x, y

def create_Reg_Dataset2():
    x = (tf.random.uniform([1, 1024], -1.0, 1.0, tf.float32, seed=1) )
    y = np.power(np.copy(x),2) - tf.random.normal([1, 1024], 0.0, 0.05,tf.float32,seed=2).numpy()
    x = np.reshape(x, (32, 32))
    y = np.reshape(y, (32, 32))

    return x, y

def create_Simple_Binary_Classifier():
    model = Sequential()
    model.add(Dense(2, activation='relu', input_dim=2))
    # model.add(Dense(1,activation='softmax',input_dim=2))
    return model
