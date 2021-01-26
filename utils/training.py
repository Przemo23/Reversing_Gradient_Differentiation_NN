#### This file contains functions used in the process of teaching models

import tensorflow as tf
import numpy as np
from optimizers.RGDOptimizer import RGDOptimizer
from pyhessian.hessian import HessianEstimators
from optimizers.ClassicMomentumOptimizer import ClassicMomentumOptimizer
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

loss_object = tf.keras.losses.MeanSquaredError()


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def loss(model, x, y, training):
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)


def train_CM(model, x_train, y_train, optimizer, epochs=10):
    # Create arrays to monitor progress

    # Training loop
    optimizer, model, loss_values = training_loop(model, x_train, y_train, None, optimizer, epochs)

    # return optimizer.v_history, optimizer.var_history
    return loss_values


def reverse_training(model, x_train, y_train, velocity, params, learning_rate, decay, epochs=10):
    hes = HessianEstimators(loss_object, model, 32)
    rgd_optimizer = RGDOptimizer(learning_rate=learning_rate, decay=decay, velocity=velocity, weights=params, hes=hes)
    rgd_optimizer.prepare_for_reverse(model.trainable_variables)

    # Create arrays to monitor progress

    rgd_optimizer, model, loss_values = training_loop(model, x_train, y_train, hes, rgd_optimizer, epochs)
    rgd_optimizer.reverse_last_step(var_list=model.trainable_variables)
    return rgd_optimizer.d_decay, rgd_optimizer.d_lr, loss_values


def training_loop(model, x_train, y_train, hes, optimizer, epochs):
    train_loss_results = []
    train_accuracy_results = []
    loss_values = []
    vars = []
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        # If hes is None it is Classic Momentum training
        # Otherwise it is reverse mode
        if hes is None:
            batch_order = range(len(x_train))
        else:
            batch_order = reversed(range(len(x_train)))

        for i in batch_order:
            x = x_train[i]
            y = y_train[i]
            loss_value, grads = grad(model, x, y)
            if hes is not None:
                hes.set_up_vars(x, y, loss_value)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            loss_values.append(loss_value)
            vars.append(model.get_weights())


        # End epoch
        train_loss_results.append(epoch_loss_avg.result())


        # print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
        #                                                             epoch_loss_avg.result(),
        #                                                             epoch_accuracy.result()))
    return optimizer, model, vars


def training_with_hypergrad(test, d_lr, d_decay, lr, decay, x, y, epochs, loss_value1,model):
    loss_values = []
    for i in range(epochs):
        lr, decay = update_hyperparams(lr, d_lr, decay, d_decay)
        CM_optimizer = ClassicMomentumOptimizer(learning_rate=lr, decay=decay)
        model.load_weights('model.h5')
        model.compile(loss='mean_squared_error', optimizer=CM_optimizer)

        train_CM(model, x, y, CM_optimizer, epochs=2)
        loss_value2 = loss(model, x.flatten(), y.flatten(), True)
        # test.assertGreaterEqual(loss_value1, loss_value2)
        print("Epoch:", i)
        print("Prev loss:", loss_value1.numpy(), ", cur loss:", loss_value2.numpy())
        print("lr:", lr)
        loss_value1 = loss_value2
        loss_values.append(loss_value2)
        d_decay, d_lr, rev_loss = reverse_training(model, x, y, CM_optimizer.v_preciserep,
                                                   params=CM_optimizer.var_preciserep,
                                                   learning_rate=CM_optimizer.l_rate, decay=CM_optimizer.decay,
                                                   epochs=2)
    plt.plot(range(epochs), loss_values )
    plt.show()
    return 0
# def get_hypergrads(d_lr, d_decay):
#     d_decay_list = []
#     d_lr_list = []
#     for key in d_lr.keys():
#         d_decay.a
#     return d_lr, d_decay


def update_hyperparams(old_lr, update_lr, old_decay, update_decay):
    new_lr = {}
    new_decay = {}
    if isinstance(old_lr, dict) and isinstance(old_decay, dict):
        for key in old_lr.keys():
            new_lr[key] = old_lr[key] - 1000000000*update_lr[key]
            new_decay[key] = old_decay[key]
    elif isinstance(old_lr, float) and isinstance(old_decay, float):
        for key in update_lr.keys():
            new_lr[key] = old_lr - 1000000000*update_lr[key]
            new_decay[key] = old_decay
    # Change to list

    # new_lr = old_lr - 10*update_lr
    # new_decay = old_decay - 10*update_decay
    return new_lr, new_decay


def create_Single_Neuron_NN(optimizer):
    model = Sequential()
    model.add(Dense(1, input_shape=[1, ]))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
