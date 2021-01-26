from utils.training import *
from optimizers.ClassicMomentumOptimizer import ClassicMomentumOptimizer
from tests.test_helpers import *
import tensorflow as tf
from pyhessian import hessian
from keras.optimizers import SGD

import matplotlib.pyplot as plt
import pandas as pd


class OptimizersTests(tf.test.TestCase):
    def setUp(self):
        super(OptimizersTests, self).setUp()

    def test_quadraticBowl(self, MAX_EPOCHS=500):
        epsilon = 0.001
        x = tf.Variable(initial_value=tf.random.uniform([1], minval=1000, maxval=1000, seed=0), name='x')
        x2 = tf.Variable(initial_value=tf.random.uniform([1], minval=1000, maxval=1000, seed=0), name='x')


        y = lambda: x * x
        y2 = lambda: x2* x2
        var = []
        vars = []

        CM_optimizer = ClassicMomentumOptimizer(learning_rate=0.02)
        for epoch in range(MAX_EPOCHS):
            CM_optimizer.minimize(y, var_list=[x])
            var.append(x.numpy()[0]*x.numpy()[0])

        plt.plot(range(MAX_EPOCHS),var,label="CM")
        plt.yscale("log")
        SGD = tf.keras.optimizers.SGD(learning_rate=0.02)
        for epoch in range(MAX_EPOCHS):
            SGD.minimize(y2, var_list=[x2])
            vars.append(x2.numpy()[0]*x2.numpy()[0])
            print(x2.numpy()[0])
        plt.plot(range(MAX_EPOCHS), vars,label="SGD")
        plt.legend()
        plt.show()


        self.assertLess(abs(x.numpy()[0]), epsilon)




    def test_reverse_one_step(self):
        x = tf.Variable(initial_value=tf.random.uniform([1], minval=1, maxval=2, seed=7), name='x')
        y = lambda: x * x
        init_x = x.numpy()
        CM_optimizer = ClassicMomentumOptimizer(learning_rate=0.2)
        CM_optimizer.minimize(y, var_list=[x])
        RGD_optimizer = RGDOptimizer(velocity=CM_optimizer.v_preciserep, weights=CM_optimizer.var_preciserep,
                                     learning_rate=CM_optimizer.l_rate, decay=CM_optimizer.decay, hes=None)
        RGD_optimizer.prepare_for_reverse([x])
        RGD_optimizer.minimize(y, var_list=[x])
        RGD_optimizer.reverse_last_step(var_list=[x])
        self.assertEqual(init_x, x.numpy())

    def test_reverse_multiple_steps(self):
        x = tf.Variable(initial_value=tf.random.uniform([1], minval=1, maxval=2, seed=5), name='x')
        y = lambda: x * x
        init_x = x.numpy()
        CM_optimizer = ClassicMomentumOptimizer(learning_rate=0.1)
        var = []

        for i in range(300):
            CM_optimizer.minimize(y, var_list=[x])
            var.append(abs(x.numpy()[0]))
        RGD_optimizer = RGDOptimizer(velocity=CM_optimizer.v_preciserep, weights=CM_optimizer.var_preciserep,
                                     learning_rate=CM_optimizer.l_rate, decay=CM_optimizer.decay, hes=None)
        RGD_optimizer.prepare_for_reverse([x])
        for i in range(300):
            RGD_optimizer.minimize(y, var_list=[x])
            var.append(abs(x.numpy()[0]))
        RGD_optimizer.reverse_last_step(var_list=[x])
        # plt.plot(range(600),var)
        # plt.yscale("log")
        # plt.show()
        self.assertEqual(init_x, x.numpy())

    def test_reversible_NN(self):
        CM_optimizer = ClassicMomentumOptimizer(learning_rate=0.1)
        x, y = create_Binary_Dataset()
        model = create_Simple_Binary_Classifier()
        model.compile(loss='binary_crossentropy',
                      optimizer=CM_optimizer,
                      metrics=['accuracy'])
        init_weights = model.get_weights()

        train_CM(model, x, y, CM_optimizer, epochs=5)
        d_decay, d_lr =  reverse_training(model, x, y, CM_optimizer.v_preciserep, params=CM_optimizer.var_preciserep,
                          learning_rate=CM_optimizer.l_rate,decay=CM_optimizer.decay, epochs=5)
        reversed_weights = model.get_weights()
        self.assertTrue(compare_weight_vectors(reversed_weights,init_weights))

    def test_regression_NN_with_tuning(self):
        lr = 0.1
        decay = 0.9
        CM_optimizer = ClassicMomentumOptimizer(learning_rate=lr, decay=decay)
        x, y = create_Reg_Dataset()


        model = create_Single_Neuron_NN(CM_optimizer)
        init_weights = model.get_weights()
        model.save_weights('model.h5')

        CM_loss = train_CM(model, x, y, CM_optimizer, epochs=2)
        loss_value1 = loss(model, x.flatten(), y.flatten(), True)
        d_decay, d_lr, rev_loss = reverse_training(model, x, y, CM_optimizer.v_preciserep,
                                                   params=CM_optimizer.var_preciserep,
                                                   learning_rate=CM_optimizer.l_rate, decay=CM_optimizer.decay,
                                                   epochs=2)
        reversed_weights = model.get_weights()
        self.assertTrue(compare_weight_vectors(reversed_weights, init_weights))
        # vars = CM_loss + rev_loss
        # w1 = [abs(var[0][0] - init_weights[0][0]) for var in vars]
        # w2 = [abs(var[1] - init_weights[1]) for var in vars]
        #
        # plt.plot(range(128), w1,label="w")
        # plt.plot(range(128), w2,label="bias")
        #
        # plt.legend()
        #
        # plt.show()

        training_with_hypergrad(self, d_lr, d_decay, lr, decay, x, y, 10, loss_value1,model)


if __name__ == '__main__':
    tf.test.main()
