import tensorflow as tf
import numpy as np
from tests.test_helpers import *
from pyhessian.hessian import HessianEstimators


class PyhessianTest(tf.test.TestCase):
    """
    Needs development
    """

    def setUp(self):
        super(PyhessianTest, self).setUp()

    def test_init(self):
        ### To be changed
        x, y = create_Binary_Dataset()
        model = create_Simple_Binary_Classifier()
        batch_size = 32
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        model.compile(loss=loss_object,
                        optimizer='SGD',
                        metrics=['accuracy'])

        hes = HessianEstimators(loss_object, model, batch_size)
        cost = loss_object(y, model(x, training=True))
        hes.set_up_vars(x, y, cost)


        est = HessianEstimators.HessianEstimator(hes,model.trainable_variables)
        hessian = est.get_H_op()
        return hessian

    # def test_get_Hv_op(self):






if __name__ == '__main__':
    tf.test.main()
