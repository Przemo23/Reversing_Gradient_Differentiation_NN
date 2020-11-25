import numpy as np
import tensorflow as tf
from utils.preciseRep import PreciseRep

RADIX_SCALE = 2 ** 52


class PreciseRepTest(tf.test.TestCase):
    def setUp(self):
        super(PreciseRepTest, self).setUp()

    def test_init(self):
        # Test if PreciseRep is correctly initialized
        arr = np.array([[0.5, 0.5], [0.5, 0.2]])
        rep = PreciseRep(arr, False)
        self.assertTrue(self.check_if_equal(arr, rep.val))
        # TO FINISH

    def test_init_2(self):
        rep = PreciseRep(np.random.rand(3, 2))
        rep2 = PreciseRep(rep.val)
        self.assertTrue(self.check_if_equal(rep.intrep, rep2.intrep))

    def test_add(self):
        # Test the precision of add operation
        arr_1 = np.array([[0.5, 0.5], [0.5, 0.2]])
        arr_2 = np.array([[0.5, 0.2], [1.2, 0.2]])
        rep = PreciseRep(arr_1, False)
        rep.add(arr_2)
        arr_1 = PreciseRep(arr_1 + arr_2)
        self.assertTrue(self.check_if_equal(rep.val, arr_1.val))

    def test_add_reversible(self):
        # Test if add operation is reversible
        arr_1 = np.array([[1.32, 0.42], [0.29, 21.3]])
        arr_2 = np.array([[12.3, 0.2], [0.31, 0.5]])
        rep = PreciseRep(arr_1)
        rep.add(arr_2)
        rep.sub(arr_2)
        self.assertTrue(self.check_if_equal(rep.val, arr_1))

    def test_sub(self):
        # Test the precision of sub operation
        arr_1 = np.array([[0.5, 0.5], [0.5, 0.2]])
        arr_2 = np.array([[0.5, 0.2], [1.2, 0.2]])
        rep = PreciseRep(arr_1, False)
        rep.sub(arr_2)
        arr_1 = PreciseRep(arr_1 - arr_2)
        self.assertTrue(self.check_if_equal(rep.val, arr_1))

    def test_mul(self):
        # Test the precision of mul operation
        arr_1 = np.array([[0.5, 0.5], [0.5, 0.2]])
        scalar = 0.462
        rep = PreciseRep(arr_1, False)
        rep.mul(scalar)
        PreciseRep(arr_1 * scalar)
        self.assertTrue(self.check_if_equal(rep.val, arr_1.avl))

    def test_mul_reversible(self):
        # Test if mul operation is reversible
        arr_1 = np.array([[0.5, 0.5], [0.5, 0.2]])
        scalar = 0.462
        rep = PreciseRep(arr_1, False)
        rep.mul(scalar)
        rep.div(scalar)
        self.assertTrue(self.check_if_equal(rep.val, arr_1))

    def test_repeatable_mul_reversible(self):
        arr_1 = np.array([[0.35, 0.235], [0.33, 0.13]])
        scalar = 0.9999
        rep = PreciseRep(arr_1, False)
        for i in range(1000):
            rep.mul(scalar)
        for i in range(1000):
            rep.div(scalar)
        self.assertTrue(self.check_if_equal(rep.val, arr_1))

    # Helper functions
    def check_if_equal(self, arr_1, arr_2):
        # Function comparing two 2-D arrays
        # epsilon = 0.00000000000001
        if arr_1.shape != arr_2.shape:
            return False
        for ix, iy in np.ndindex(arr_1.shape):
            if abs(arr_1[ix][iy] - arr_2[ix][iy]) > 0.0:
                return False
        return True


if __name__ == '__main__':
    tf.test.main()
