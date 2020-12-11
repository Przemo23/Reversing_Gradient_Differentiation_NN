import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import state_ops
from utils.preciseRep import PreciseRep
from pyhessian.hessian import HessianEstimators


def prepare_for_reverse(weights, velocity, learning_rate):
    # This function manually undoes the last weights update.
    # It is necessary, since we need to calculate the gradient
    # concerning the weights from one state before the one
    # we are in.
    for w in weights:
        state_ops.assign(w, w - learning_rate * velocity[w.ref()].val)


class RGDOptimizer(keras.optimizers.Optimizer):

    def __init__(self, velocity, hes, learning_rate=0.1, decay=0.9, name="RGDOptimizer"):
        """
        :param velocity: The velocity matrix of the Classic Momentum Optimizer after it finished optimizing the model
        :param hes: HessianEstimators objects containing necessary params to calculate hessians
        :param learning_rate: Learning rate controls the impact of the velocity on weights. Has to be >0.
        :param decay: Decay controls the growth of the velocity. It has to be inside (0;1)
        :param name: The name of the optimizer
        """
        super().__init__(name)
        self._init_hyperparams(learning_rate, decay, velocity, hes)

    def _init_hyperparams(self, learning_rate, decay, velocity, hes):
        """
        :var self.var_preciserep: A dictionary containing the precise representations of weights for each layer
        :var self.v_preciserep: A dictionary containing the precise representations of velocities for each layer
        """
        self.learning_rate = learning_rate
        self.decay = decay
        self.init_dict = {}
        self.var_preciserep = {}
        self.v_preciserep = velocity
        self.hes = hes
        self.d_lr = {}
        self.d_decay = {}

    def assign_to_slots(self, var_list):
        # Initialize the slots and create the precise representations of weights
        for var in var_list:
            self.var_preciserep[var.ref()] = PreciseRep(var.numpy())
            self.d_decay[var.ref()] = 0.0
            self.d_lr[var.ref()] = 0.0
            self.get_slot(var, "d_v").assign(np.zeros(var.shape))
            # self.get_slot(var, "d_teta").assign(np.zeros(var.shape))

    def create_init_dict(self, var_list):
        for var in var_list:
            self.init_dict[var.ref()] = True;

    def _create_slots(self, var_list):
        # _create_slots is supposed to be used only for creating slots for the layers of the model
        # However as it is called in every iteration of the optimizing algorithm and has access to
        # the var_list, it is used here to create the precise representations of  weights
        # for each layer in var_list

        for var in var_list:
            self.add_slot(var, "d_v")
            # self.add_slot(var, "d_teta")
            self.add_slot(var, "d_x")

            self.hes.estimators[var.ref()] = HessianEstimators.HessianEstimator(self.hes, [var])
        if var_list[0].ref() not in self.var_preciserep.keys():
            self.assign_to_slots(var_list)
            self.create_init_dict(var_list)

    def _resource_apply_dense(self, grad, var):
        lr = self.learning_rate
        decay = self.decay

        # Used for inintialization
        if self.init_dict[var.ref()]:
            self.init_dict[var.ref()] = False
            self.get_slot(var, "d_x").assign(grad)

        # Assign variables from slots and class variables
        x = self.var_preciserep[var.ref()]
        v = self.v_preciserep[var.ref()]
        dv = self.get_slot(var, 'd_v')
        dx = self.get_slot(var, 'd_x')

        # It needs to be double transposed, so it remains in its initial shape
        dx_identity = tf.reshape(tf.identity(dx),[-1])
        dx_numpy = tf.transpose(dx_identity).numpy()
        v_numpy = v.val.ravel()
        self.d_lr[var.ref()] = np.dot(dx_numpy, v_numpy)

        # Revert the CM optimizers steps
        v.add(grad.numpy() * (1 - decay))
        v.div(decay)
        x.sub(v.val * lr)

        # Assign the new values of weights and velocities
        state_ops.assign(var, x.val)
        self.var_preciserep[var.ref()] = x
        self.v_preciserep[var.ref()] = v

        # Calculate the gradient of the learning trajectory
        state_ops.assign_add(dv, lr * dx)
        dv_identity = tf.reshape(tf.identity(dv), [-1])
        dv_numpy = tf.transpose(dv_identity).numpy()
        self.d_decay = np.dot(dv_numpy, (grad.numpy()+v.val).ravel())
        new_dx = (1 - decay) * self.hes.estimators[var.ref()].get_Hv_op(tf.transpose(dv_identity))
        state_ops.assign_sub(dx, tf.reshape(new_dx, var.shape))
        state_ops.assign(dv, dv * decay)

    def _reverse_last_step(self, var_list):
        # As we called prepare_for_reverse() before starting the optimization and then we called
        # _resource_apply_dense() N times, the weights have been actualized one time to many.
        # This function reverts the last update, so the weights are updated only N times in the end.

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
