from keras.layers import Dense
from keras.models import Sequential
from utils.training import *
from optimizers.ClassicMomentumOptimizer import ClassicMomentumOptimizer
from keras.utils.np_utils import to_categorical
from tests.optimizers_tests import OptimizersTests
from tests.preciseRep_test import PreciseRepTest
from tuning_example import tuning_example_reg
from tuning_example import tuning_example_reg2

optimizers_tests = OptimizersTests()
preciserep_tests = PreciseRepTest()
optimizers_tests.run_tests()
preciserep_tests.run_tests()
tuning_example_reg()
tuning_example_reg2()