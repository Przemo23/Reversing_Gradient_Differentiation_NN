import numpy as np

RADIX_SCALE = 2 ** 52


def float_to_rational(a):
    assert np.all(a > 0.0)
    d = int(2 ** 16 / np.fix(a + 1).astype(int))  # Uglier than it used to be: np.int(a + 1)
    n = np.fix(a * d + 1).astype(int)
    return n, d


def float_to_intrep(x):
    return (x * RADIX_SCALE).astype(np.int64)


class PreciseRep(object):
    def __init__(self, val, from_intrep=False):
        if from_intrep:
            self.intrep = val
        else:
            self.intrep = float_to_intrep(val)
        self.aux = BitStore(val.shape)

    def add(self, A):
        """Reversible addition of vector or scalar A."""
        self.intrep += float_to_intrep(A)
        return self

    def sub(self, A):
        self.add(-A)
        return self

    def rational_mul(self, n, d):
        self.aux.push(self.intrep % d, d)  # Store remainder bits externally
        self.intrep = self.intrep // d
        self.intrep *= n  # Multiply by numerator
        self.intrep += self.aux.pop(n)

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
        return self.intrep.astype(np.float64) / RADIX_SCALE


#
# class LongIntArray(object):
#     """Behaves like np.array([0L] * length, dtype=object) but faster."""
#
#     def __init__(self, shape):
#         self.val = []
#         self.nbits = 0
#         self.grow()
#         self.shape = shape
#
#     def grow(self):
#         self.val.append(np.zeros(self.shape, dtype=np.int32))
#         self.nbits += 32
#

class BitStore(object):
    """Efficiently stores information with non-integer number of bits (up to 16)."""

    def __init__(self, shape):
        # Use an array of Python 'long' ints which conveniently grow
        # as large as necessary. It's about 50X slower though...

        self.store = np.zeros(shape, dtype=int)

    def push(self, N, M):
        """Stores integer N, given that 0 <= N < M"""
        assert np.all(M <= 2 ** 16)
        self.store *= M
        self.store += N

    def pop(self, M):
        """Retrieves the last integer stored."""
        N = self.store % M
        self.store = self.store / M
        return N.astype(int)
