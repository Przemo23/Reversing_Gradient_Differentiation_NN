from utils.preciseRep import PreciseRep
from tensorflow import keras
from tensorflow.python.ops import state_ops

import numpy as np


class ClassicMomentumOptimizer(keras.optimizers.Optimizer):

    def __init__(self, learning_rate=0.1, decay=0.9, name="ClassicMomentumOptimizer"):
        super().__init__(name)
        self._decay = decay
        self._lr = learning_rate
        self.var_preciserep = {}
        self.v_preciserep = {}

    def _prepare(self, var_list):
        if var_list[0].ref() not in self.var_preciserep.keys():
            for var in var_list:
                self.var_preciserep[var.ref()] = PreciseRep(var.numpy())
                self.v_preciserep[var.ref()] = PreciseRep(np.zeros(var.shape))

    def _resource_apply_dense(self, grad, var):
        x = self.var_preciserep[var.ref()]
        v = self.v_preciserep[var.ref()]
        g = PreciseRep(grad.numpy())
        lr = self._lr
        decay = self._decay

        # get current velocity
        # update position
        g.mul(1-decay)
        v.mul(decay)
        v.sub(g.val)
        x.add(v.val*lr)

        self.var_preciserep[var.ref()] = x
        state_ops.assign(var,x.val)
        self.v_preciserep[var.ref()] = v

    #   return control_flow_ops.group(*[var_update, v_t])
    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._lr,
            "decay": self._decay,
        }
