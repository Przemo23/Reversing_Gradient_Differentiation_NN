import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import state_ops
from utils.preciseRep import PreciseRep
from pyhessian.pyhessian import HessianEstimators


def prepare_for_reverse(weights, velocity, learning_rate):

    for w in weights:
        state_ops.assign(w,w-learning_rate* velocity[w.ref()].val)


class RGDOptimizer(keras.optimizers.Optimizer):

    def __init__(self, velocity, hes, learning_rate=0.1, decay=0.9, name="RGDOptimizer"):
        super().__init__(name)
        self._init_hyperparams(learning_rate, decay, velocity, hes)

    def _init_hyperparams(self, learning_rate, decay, velocity, hes):
        self.learning_rate = learning_rate
        self.decay = decay
        self.init_dict = {}
        self.var_preciserep = {}
        self.v_preciserep = velocity
        self.hes = hes

    def assign_to_slots(self, var_list):
        for var in var_list:
            self.var_preciserep[var.ref()] = PreciseRep(var.numpy())

            self.get_slot(var, "d_v").assign(np.zeros(var.shape))
            self.get_slot(var, "d_teta").assign(np.zeros(var.shape))
            self.get_slot(var, "d_lr").assign(np.zeros(var.shape))
            self.get_slot(var, "d_decay").assign(np.zeros(var.shape))

    def create_init_dict(self, var_list):
        for var in var_list:
            self.init_dict[var.name] = True;

    def _create_slots(self, var_list):
        # self.velocity.shape()
        # for var in var_list:
        for var in var_list:
            self.add_slot(var, "d_v")
            self.add_slot(var, "d_lr")
            self.add_slot(var, "d_teta")
            self.add_slot(var, "d_decay")
            self.add_slot(var, "d_x")
            self.add_slot(var, "var_precise_rep")
            self.add_slot(var, "v_precise_rep")

            self.hes.estimators[var.ref()] = HessianEstimators.HessianEstimator(self.hes, var.numpy())
        if var_list[0].ref() not in self.var_preciserep.keys():
            self.assign_to_slots(var_list)
            self.create_init_dict(var_list)
        # self.update_hes_estimators(var_list)

    def _resource_apply_dense(self, grad, var):
        lr = self.learning_rate
        decay = self.decay

        #Used for inintialization
        if self.init_dict[var.name]:
            self.init_dict[var.name] = False
            self.get_slot(var, "d_x").assign(grad)

        x = self.var_preciserep[var.ref()]
        v = self.v_preciserep[var.ref()]
        dv = self.get_slot(var,'d_v')
        dx = self.get_slot(var,'d_x')
        d_decay = self.get_slot(var,'d_decay')



        state_ops.assign(self.get_slot(var,"d_lr"), tf.transpose(dx) * v.val)
        tf.transpose(dx)
        v.add(grad.numpy()*(1-decay))
        v.div(decay)
        x.sub(v.val * lr)

        state_ops.assign(var, x.val)
        self.var_preciserep[var._ref()] = x
        self.v_preciserep[var.ref()] = v

        state_ops.assign_add(dv,lr*dx)
        state_ops.assign(d_decay,tf.transpose(dv)*(grad.numpy()+v.val))
        state_ops.assign_sub(dx,(1-decay)*self.hes.estimators[var].get_Hv_op(dv))
        state_ops.assign(dv,dv*decay)


    def _reverse_last_step(self, var_list):
        for var in var_list:
            self.var_preciserep[var.ref()].add(self.v_preciserep[var.ref()].val * self.learning_rate)
            state_ops.assign(var, self.var_preciserep[var.ref()].val)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self.learning_rate,
            "decay": self.decay,
        }
