from utils.training import *
from optimizers.ClassicMomentumOptimizer import ClassicMomentumOptimizer
from tests.test_helpers import *
import pandas as pd


class OptimizersTests(tf.test.TestCase):
    def setUp(self):
        super(OptimizersTests, self).setUp()
        # self.CM_optimizer = ClassicMomentumOptimizer(learning_rate=0.1)

    def test_quadraticBowl(self, MAX_EPOCHS=500):
        # tf.compat.v1.disable_eager_execution()
        epsilon = 0.001
        x = tf.Variable(initial_value=tf.random.uniform([1], minval=1000, maxval=2000, seed=0), name='x')
        y = lambda: x * x

        CM_optimizer = ClassicMomentumOptimizer(learning_rate=0.1)
        for epoch in range(MAX_EPOCHS):
            CM_optimizer.minimize(y, var_list=[x])
        self.assertLess(abs(x), epsilon)

    def test_reverse_one_step(self):
        x = tf.Variable(initial_value=tf.random.uniform([1], minval=1, maxval=2, seed=7), name='x')
        y = lambda: x * x
        init_x = x.numpy()
        CM_optimizer = ClassicMomentumOptimizer(learning_rate=0.2)
        CM_optimizer.minimize(y, var_list=[x])
        RGD_optimizer = RGDOptimizer(velocity=CM_optimizer.v_preciserep, weights=CM_optimizer.var_preciserep,
                                     learning_rate=CM_optimizer.l_rate,decay=CM_optimizer.decay, hes=None)
        RGD_optimizer.prepare_for_reverse([x])
        RGD_optimizer.minimize(y, var_list=[x])
        RGD_optimizer.reverse_last_step(var_list=[x])
        self.assertEqual(init_x, x.numpy())

    def test_reverse_multiple_steps(self):
        x = tf.Variable(initial_value=tf.random.uniform([1], minval=1, maxval=2, seed=5), name='x')
        y = lambda: x * x
        init_x = x.numpy()
        CM_optimizer = ClassicMomentumOptimizer(learning_rate=0.1)

        for i in range(300):
            CM_optimizer.minimize(y, var_list=[x])
        # prepare_for_reverse([x], velocity=self.CM_optimizer.v_preciserep, learning_rate=0.2)
        RGD_optimizer = RGDOptimizer(velocity=CM_optimizer.v_preciserep, weights=CM_optimizer.var_preciserep,
                                     learning_rate=CM_optimizer.l_rate,decay=CM_optimizer.decay, hes=None)
        RGD_optimizer.prepare_for_reverse([x])
        for i in range(300):
            RGD_optimizer.minimize(y, var_list=[x])
        RGD_optimizer.reverse_last_step(var_list=[x])
        self.assertEqual(init_x, x.numpy())

    # def test_reversible_NN(self):
    #     CM_optimizer = ClassicMomentumOptimizer(learning_rate=0.1)
    #     x, y = create_Binary_Dataset()
    #     model = create_Simple_Binary_Classifier()
    #     model.compile(loss='binary_crossentropy',
    #                   optimizer=CM_optimizer,
    #                   metrics=['accuracy'])
    #     init_weights = model.get_weights()
    #
    #     train_CM(model, x, y, CM_optimizer, epochs=5)
    #     d_decay, d_lr =  reverse_training(model, x, y, CM_optimizer.v_preciserep, params=CM_optimizer.var_preciserep,
    #                       learning_rate=CM_optimizer.l_rate,decay=CM_optimizer.decay, epochs=5)
    #     reversed_weights = model.get_weights()
    #     self.assertTrue(compare_weight_vectors(reversed_weights,init_weights))

    def test_regression_NN_with_tuning(self):
        lr = 0.1
        decay = 0.9
        CM_optimizer = ClassicMomentumOptimizer(learning_rate=lr, decay=decay)
        x, y = create_Reg_Dataset()
        model = Sequential()
        model.add(Dense(1, input_shape=[1, ]))
        model.compile(loss='mean_squared_error', optimizer=CM_optimizer)
        init_weights = model.get_weights()

        train_CM(model, x, y, CM_optimizer, epochs=5)
        loss_value1 = loss(model, x[0], y[0], True)
        d_decay, d_lr = reverse_training(model, x, y, CM_optimizer.v_preciserep, params=CM_optimizer.var_preciserep,
                                         learning_rate=CM_optimizer.l_rate, decay=CM_optimizer.decay, epochs=5)
        reversed_weights = model.get_weights()
        self.assertTrue(compare_weight_vectors(reversed_weights, init_weights))

        new_lr, new_decay = update_hyperparams(lr,d_lr,decay,d_decay)
        CM_optimizer = ClassicMomentumOptimizer(learning_rate=new_lr,decay=new_decay)
        train_CM(model, x, y, CM_optimizer, epochs=5)
        loss_value2 = loss(model, x[0], y[0], True)
        self.assertGreater(loss_value1,loss_value2)



if __name__ == '__main__':
    tf.test.main()
