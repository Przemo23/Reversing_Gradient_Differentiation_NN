#### This file contains functions used in the process of teaching models
#### and testing the optimizers

import tensorflow as tf
import numpy as np
from optimizers.RGDOptimizer import RGDOptimizer
from pyhessian.hessian import HessianEstimators

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
    train_loss_results = []
    train_accuracy_results = []

    # Training loop
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for i in range(len(x_train)):
            x = x_train[i]
            y = y_train[i]
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y, model(x, training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

    # return optimizer.v_history, optimizer.var_history


def reverse_training(model, x_train, y_train, velocity, params, epochs=10):
    hes = HessianEstimators(loss_object, model, 32)
    rgd_optimizer = RGDOptimizer(velocity, params, hes)
    rgd_optimizer.prepare_for_reverse(model.trainable_variables)

    # Create arrays to monitor progress
    train_loss_results = []
    train_accuracy_results = []

    # Training loop
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for i in reversed(range(len(x_train))):
            x = x_train[i]
            y = y_train[i]
            loss_value, grads = grad(model, x, y)
            hes.set_up_vars(x, y, loss_value)
            rgd_optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, model(x, training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

    rgd_optimizer._reverse_last_step(var_list=model.trainable_variables)
    return rgd_optimizer.d_decay, rgd_optimizer.d_lr
