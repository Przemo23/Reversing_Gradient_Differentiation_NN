from utils.preciseRep import PreciseRep
from tensorflow import keras
from tensorflow.python.ops import state_ops
from utils.preciseRep import list_operation

import numpy as np


class ClassicMomentumOptimizer(keras.optimizers.Optimizer):

    def __init__(self, learning_rate=0.1, decay=0.9, name="ClassicMomentumOptimizer"):
        """
        :param learning_rate: Learning rate controls the impact of the velocity on weights. Has to be >0.
        :param decay: Decay controls the growth of the velocity. It has to be inside (0;1)
        :param name: Name of the params

        :var self.var_preciserep: A dictionary containing the precise representations of weights for each layer
        :var self.v_preciserep: A dictionary containing the precise representations of velocities for each layer
        """
        super().__init__(name)
        self.decay_temp = decay
        self.lr_temp = learning_rate
        self.var_preciserep = {}
        self.v_preciserep = {}
        self.l_rate = {}
        self.decay = {}
        # self.var_history = {}
        # self.v_history = {}

    def _prepare(self, var_list):

        if var_list[0].ref() not in self.var_preciserep.keys():
            if isinstance(self.lr_temp, dict):
                for var in var_list:
                    self.l_rate[var.ref()] = self.lr_temp[var.ref()]
                    self.decay[var.ref()] = self.decay_temp[var.ref()]
                    self.var_preciserep[var.ref()] = PreciseRep(var.numpy().ravel().tolist())
                    self.v_preciserep[var.ref()] = PreciseRep(np.zeros(var.shape).ravel().tolist())


            elif isinstance(self.lr_temp, float):
                for var in var_list:
                    self.l_rate[var.ref()] = self.lr_temp
                    self.var_preciserep[var.ref()] = PreciseRep(var.numpy().ravel().tolist())
                    self.v_preciserep[var.ref()] = PreciseRep(np.zeros(var.shape).ravel().tolist())
                    self.decay[var.ref()] = self.decay_temp
                # self.var_history[var.ref()] = []
                # self.v_history[var.ref()] = []

    def _resource_apply_dense(self, grad, var):
        x = self.var_preciserep[var.ref()]
        v = self.v_preciserep[var.ref()]
        # self.v_history[var.ref()].append(v.val)
        # self.var_history[var.ref()].append(x.val)

        lr = self.l_rate[var.ref()]
        decay = self.decay[var.ref()]

        # get current velocity
        # update position
        # gradP = PreciseRep(np.array([1 - decay]))
        # gradP.mul_scalar_matrix(grad.numpy())

        v.mul([decay])
        v.sub((grad.numpy() * (1 - decay)).ravel().tolist())
        x.add(list_operation(v.val, '*', [lr]))

        self.var_preciserep[var.ref()] = x
        state_ops.assign(var, np.array(x.val).reshape(var.shape))
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
