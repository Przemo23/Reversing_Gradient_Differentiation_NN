import numpy as np
import tensorflow as tf
from utils.preciseRep import PreciseRep

RADIX_SCALE = 2 ** 52


class PreciseRepTest(tf.test.TestCase):
    def setUp(self):
        super(PreciseRepTest, self).setUp()

    def test_init(self):
        # Test if PreciseRep is correctly initialized
        arr = [0.5, 0.5, 0.5, 0.2]
        rep = PreciseRep(arr, False)
        self.assertTrue(self.check_if_equal(arr, rep.val))
        # TO FINISH

    def test_init_2(self):
        rep = PreciseRep(np.random.rand(6).tolist())
        rep2 = PreciseRep(rep.val)
        self.assertTrue(self.check_if_equal(rep.intrep, rep2.intrep))

    def test_add(self):
        # Test the precision of add operation
        arr_1 = [0.5, 0.5, 0.5, 0.2]
        arr_2 = [0.5, 0.2, 1.2, 0.2]
        rep = PreciseRep(arr_1, False)
        rep.add(arr_2)
        arr_1 = [arr_1[i] + arr_2[i] for i in range(len(arr_1))]
        self.assertTrue(self.check_if_equal(rep.val, arr_1))

    def test_add_reversible(self):
        # Test if add operation is reversible
        arr_1 = [1.32, 0.42, 0.29, 21.3]
        arr_2 = [12.3, 0.2, 0.31, 0.5]
        rep = PreciseRep(arr_1)
        rep.add(arr_2)
        rep.sub(arr_2)
        self.assertTrue(self.check_if_equal(rep.val, arr_1))

    def test_sub(self):
        # Test the precision of sub operation
        arr_1 = [0.5, 0.5, 0.5, 0.2]
        arr_2 = [0.5, 0.2, 1.2, 0.2]
        rep = PreciseRep(arr_1, False)
        rep.sub(arr_2)
        arr_1 = [arr_1[i] - arr_2[i] for i in range(len(arr_1))]
        self.assertTrue(self.check_if_equal(rep.val, arr_1))

    def test_mul(self):
        # Test the precision of mul operation
        arr_1 = [0.5, 0.5, 0.5, 0.2]
        scalar = [0.462]
        rep = PreciseRep(arr_1, False)
        rep.mul(scalar)
        arr_1 = [arr_1[i] * scalar[0] for i in range(len(arr_1))]
        self.assertTrue(self.check_if_equal(rep.val, arr_1))

    def test_mul_reversible(self):
        # Test if mul operation is reversible
        arr_1 = [0.5, 0.5, 0.5, 0.2]
        scalar = [0.462]
        rep = PreciseRep(arr_1, False)
        rep.mul(scalar)
        rep.div(scalar)
        self.assertTrue(self.check_if_equal(rep.val, arr_1))

    def test_repeatable_mul_reversible(self):
        arr_1 = [0.35, 0.235, 0.33, 0.13]
        scalar = [0.9999]
        rep = PreciseRep(arr_1, False)
        for i in range(200):
            rep.mul(scalar)
        for i in range(200):
            rep.div(scalar)
        self.assertTrue(self.check_if_equal(rep.val, arr_1))
    def test_scalar_mul_val(self):
        iters = 19
        rep = PreciseRep([1.0000001])
        repvals = []
        for i in range(iters):
            repvals.append(rep.val)
            rep.mul(rep.val)
        for i in reversed(range(iters)):
            rep.div(repvals[i])
        self.assertEqual(rep.val, [1.0000001])

    # def test_matrix_mul_val(self):
    #     iters = 100
    #     rep = PreciseRep(np.full(3, 1.01).tolist())
    #     repvals = []
    #
    def test_double_mul_div(self):
        iters = 10000
        scalars = (0.05 * np.random.randn(iters) + 1).tolist()
        rep1 = PreciseRep([1.0])
        rep2 = PreciseRep([0.9])
        forward = []
        backward = []

        for i in range(iters):
            forward.append((rep2.val, rep1.val))
            rep1.mul([scalars[i]])
            rep2.mul(rep1.val)

        for i in range(iters):
            rep2.div(rep1.val)
            rep1.div([scalars[iters - 1 - i]])
            backward.insert(0,(rep2.val,rep1.val))

        # errors = [forward[i][0]-backward[i][0] for i in range(iters)]

        self.assertTrue(self.check_if_equal(rep1.val, [1.0]))
        self.assertTrue(self.check_if_equal(rep2.val, [0.9]))
    #



    # def test_gradient_mul(self):
    #     grad = 0.05*np.random.randn(10000) + 1
    #     decay = PreciseRep(np.array([1.0]), False)
    #     arr_1 = np.array([[0.35, 0.235], [0.33, 0.13]])
    #     rep = PreciseRep(arr_1, False)
    #     for j in range(100):
    #         for i in range(100):
    #             decay.mul(grad[j*100+i])
    #             rep.mul(decay.val)
    #         for i in range(100):
    #             rep.div(decay.val)
    #             decay.div(grad[(j+1)*100-1-i])
    #
    #     self.assertTrue(self.check_if_equal(rep.val, arr_1))



    # Helper functions
    def check_if_equal(self, list_1, list_2):
        epsilon = 0.00001
        if len(list_1) != len(list_2):
            return False

        for i in range(len(list_1)):
            if abs(list_1[i] - list_2[i]) >= epsilon:
                return False
        return True
        # for ix, iy in np.ndindex(arr_1.shape):
        #     if arr_1[ix][iy] - arr_2[ix][iy] >= 0.000001:
        #         return False
        # return True


if __name__ == '__main__':
    tf.test.main()
