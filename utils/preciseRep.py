import numpy as np
import operator

RADIX_SCALE = 2 ** 52



def list_operation(arg1, op, arg2):
    ops = {'//': operator.floordiv,
           '%': operator.mod,
           '+': operator.add,
           '*': operator.mul}
    if len(arg2) == 1:
        return [ops[op](x, arg2[0]) for x in arg1]
    else:
        return [ops[op](arg1[i], arg2[i]) for i in range(len(arg1))]


def float_to_rational(a):
    assert all(x > 0.0 for x in a)
    d = [int(2 ** 16 / int(x + 1)) for x in a]
    n = [int(a[i] * d[i] + 1) for i in range(len(a))]
    return n, d

# def float_to_rational(a):
#     assert np.all(a > 0.0)
#     d = int(2 ** 16 / int(a + 1))  # Uglier than it used to be: np.int(a + 1)
#     n = int(a * d + 1)
#     return n, d



def float_to_intrep(x):
    return [int(var * RADIX_SCALE) for var in x]


class PreciseRep(object):
    def __init__(self, val, from_intrep=False):
        if from_intrep:
            self.intrep = val
        else:
            self.intrep = float_to_intrep(val)
        self.aux = BitStore(len(val))

    def add(self, A):
        """Reversible addition of vector or scalar A."""
        if len(A) != 1 and len(A) != len(self.intrep):
            raise Exception("A is a vector and the sizes are not matching.")

        A = float_to_intrep(A)
        self.intrep = list_operation(self.intrep, '+', A)
        return self

    def sub(self, A):
        self.add([-x for x in A])
        return self

    def rational_mul(self, n, d):
        self.aux.push(list_operation(self.intrep, '%', d), d)  # Store remainder bits externally
        self.intrep = list_operation(self.intrep, '//', d)
        self.intrep = list_operation(self.intrep, '*', n)
        self.intrep = list_operation(self.intrep, '+', self.aux.pop(n))

    def mul(self, a):
        n, d = float_to_rational(a)
        self.rational_mul(n, d)
        return self

    def div(self, a):
        n, d = float_to_rational(a)
        self.rational_mul(d, n)
        return self

    @property
    def val(self):
        return [float(x) / RADIX_SCALE for x in self.intrep]

class BitStore(object):
    """Efficiently stores information with non-integer number of bits (up to 16)."""

    def __init__(self, size):
        self.store = [0] * size

    def push(self, N, M):
        """Stores integer N, given that 0 <= N < M"""
        assert all(m <= 2 ** 16 for m in M)
        self.store = list_operation(self.store, '*', M)
        self.store = list_operation(self.store, '+', N)

        # assert np.all(M <= 2 ** 16)
        # self.store = M
        # self.store += N

    def pop(self, M):
        """Retrieves the last integer stored."""
        N = list_operation(self.store, '%', M)
        self.store = list_operation(self.store, '//', M)

        # N = self.store % M
        # self.store = self.store // M
        # return int(N)

        return N
