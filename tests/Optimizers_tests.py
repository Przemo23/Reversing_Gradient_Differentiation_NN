from utils.training import *
from optimizers.ClassicMomentumOptimizer import ClassicMomentumOptimizer
from tests.test_helpers import *


class OptimizersTests(tf.test.TestCase):
    def setUp(self):
        super(OptimizersTests, self).setUp()
        self.CM_optimizer = ClassicMomentumOptimizer(learning_rate=0.2)

    def test_quadraticBowl(self, MAX_EPOCHS=500):
        # tf.compat.v1.disable_eager_execution()
        epsilon = 0.001
        x = tf.Variable(initial_value=tf.random.uniform([1], minval=1000, maxval=2000, seed=0), name='x')
        y = lambda: x * x

        for epoch in range(MAX_EPOCHS):
            self.CM_optimizer.minimize(y, var_list=[x])
        self.assertLess(abs(x), epsilon)

    # def test_reverse_one_step(self):
    #     x = tf.Variable(initial_value=tf.random.uniform([1], minval=1, maxval=2, seed=7), name='x')
    #     y = lambda: x * x
    #     init_x = x.numpy()
    #     self.CM_optimizer.minimize(y, var_list=[x])
    #     prepare_for_reverse([x], velocity=self.CM_optimizer.v_preciserep, learning_rate=0.2)
    #     RGD_optimizer = RGDOptimizer(velocity=self.CM_optimizer.v_preciserep, learning_rate=0.2)
    #     RGD_optimizer.minimize(y, var_list=[x])
    #     RGD_optimizer._reverse_last_step(var_list=[x])
    #     self.assertEqual(init_x, x.numpy())

    def test_reverse_multiple_steps(self):
        x = tf.Variable(initial_value=tf.random.uniform([1], minval=1, maxval=2, seed=5), name='x')
        y = lambda: x * x
        init_x = x.numpy()
        for i in range(30000):
            self.CM_optimizer.minimize(y, var_list=[x])
        prepare_for_reverse([x], velocity=self.CM_optimizer.v_preciserep, learning_rate=0.2)
        RGD_optimizer = RGDOptimizer(velocity=self.CM_optimizer.v_preciserep, learning_rate=0.2,hes=None)
        for i in range(30000):
            RGD_optimizer.minimize(y, var_list=[x])
        RGD_optimizer._reverse_last_step(var_list=[x])
        self.assertEqual(init_x, x.numpy())

    # def test_reversible_NN(self):
    #     x, y = create_Binary_Dataset()
    #     model = create_Simple_Binary_Classifier()
    #     model.compile(loss='binary_crossentropy',
    #                   optimizer=self.CM_optimizer,
    #                   metrics=['accuracy'])
    #     init_weights = model.get_weights()
    #
    #     train_CM(model, x, y, self.CM_optimizer, epochs=4)
    #     train_weights = model.get_weights()
    #     reverse_training(model, x, y, self.CM_optimizer.v_preciserep, epochs=4)
    #     reversed_weights = model.get_weights()
    #     # self.assertEqual(np.array(init_weights), np.array(reversed_weights))


if __name__ == '__main__':
    tf.test.main()
