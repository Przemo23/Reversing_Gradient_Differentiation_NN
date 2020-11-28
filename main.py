from keras.layers import Dense
from keras.models import Sequential
from utils.training import *
from optimizers.ClassicMomentumOptimizer import ClassicMomentumOptimizer
from keras.utils.np_utils import to_categorical


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train[:40000]
x_test = x_test[:8000]
y_train = y_train[:40000]
y_test = y_test[:8000]
x_train = np.reshape(x_train, (1250,32, 3072))
x_test = np.reshape(x_test, (250,32, 3072))
y_train = np.reshape(y_train, (1250,32))
y_test = np.reshape(y_test, (250,32))
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
x_train = x_train[:100]
x_test = x_test[:100]

x_train = x_train / 255
x_test = x_test / 255

# cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=3072, name='layer_1'))
model.add(Dense(256, activation='relu', name='layer_2'))
model.add(Dense(10, activation='softmax', name='layer_out'))

classic_momentum_optimizer = ClassicMomentumOptimizer()
model.compile(optimizer=classic_momentum_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


NUM_EPOCHS = 3
BATCH_SIZE = 32

train_CM(model,x_train,y_train,classic_momentum_optimizer,epochs=NUM_EPOCHS)

reverse_training(model,x_train,y_train,classic_momentum_optimizer.v_preciserep,
                 epochs=NUM_EPOCHS)
