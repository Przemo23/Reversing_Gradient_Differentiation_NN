import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential


def create_Test_Binary_Dataset():
    x = tf.random.normal([512, 2], 2, 1.0, tf.float32, seed=1).numpy()
    x = np.concatenate((x, tf.random.normal([512, 2], -2, 1.0, tf.float32, seed=1).numpy()), axis=0)
    y = np.concatenate((np.ones(512),np.zeros(512)),axis=0)
    x,y = shuffle(x,y,random_state=0)
    # Create batches
    x = np.reshape(x,(32,32,2))
    y = np.reshape(y,(32,32)).astype('int64')
    y = to_categorical(y,num_classes=2)

    return x, y

def create_Simple_Binary_Classifier():
    model = Sequential()
    # 1 neuron. Basically fit a line.

    model.add(Dense(4,activation='relu',input_dim=2))
    model.add(Dense(2,activation='softmax',input_dim=2))
    return model


