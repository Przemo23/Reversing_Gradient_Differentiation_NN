"""
    hessian.py - Hessian Matrix Estimator
 
    Copyright (c) 2018-2019 by Geir K. Nilsen (geir.kjetil.nilsen@gmail.com)
    and the University of Bergen.
 
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
 
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import tensorflow as tf
import numpy as np


class HessianEstimators():
    def __init__(self, cost_fun, model_fun,
                 batch_size_G):
        self.cost_fun = cost_fun
        self.model_fun = model_fun
        self.batch_size_G = batch_size_G
        self.estimators = {}

    def set_up_vars(self, x, y, loss_val):
        self.x = x
        self.y = y
        self.loss_val = loss_val

    class HessianEstimator(object):
        """ Implements a Hessian matrix estimator

        Attributes:
            cost_fun: Cost function (function)
            cost: Cost function output (tensor)
            model_fun: Model function (function)
            params: List of model parameters (list of tensor(s))
            P: Total number of model parameters (int)
            x: Model input (tensor)
            y: Model output (tensor)
            batch_size_G: Batch size used to estimate Hessian OPG approximation

        """

        def __init__(self, hessianEstimators, params):
            """
            Args:
                cost_fun(y, yhat_logits, params): Cost function (function)
                    Args:
                        y: Labels (tensor)
                        yhat_logits: Model output in logits (tensor)
                        params: List of model parameter(s) \
                                (List of tensor(s)) [Can be used for
                                                     regularization
                                                     purposes]
                    Returns:
                        cost: Cost function output (tensor)

                cost: Cost function output (tensor)
                model_fun(x, params): Model function (function),
                    Args:
                        X: Model input (tensor)
                        params:  List of model parameter(s) \
                                 (List of tensor(s))

                    Returns:
                        y: Model output (tensor)
                params: List of model parameters (list of tensor(s))
                X: Model input (tensor)
                y: Model output (tensor)
                batch_size_G: Batch size used to estimate Hessian OPG
                              approximation (int)
            """
            self.cost_fun = hessianEstimators.cost_fun
            self.cost = hessianEstimators.loss_val
            self.model_fun = hessianEstimators.model_fun
            self.params = params
            self.P = self.flatten(self.params).get_shape().as_list()[0]
            self.x = hessianEstimators.x
            self.y = hessianEstimators.y
            self.batch_size_G = hessianEstimators.batch_size_G

        # def set_params(self, cost, x, y, params):
        #     self.x = x
        #     self.y = y
        #     self.cost = cost
        #     self.params = params

        def flatten(self, params):
            """
            Flattens the list of tensor(s) into a 1D tensor

            Args:
                params: List of model parameters (List of tensor(s))

            Returns:
                A flattened 1D tensor
            """
            flattened_params = [tf.reshape(_params, [-1]) for _params in params]
            return tf.concat(flattened_params,0)

        def get_Hv_op(self, v):
            """
            Implements a Hessian vector product estimator Hv op defined as the
            matrix multiplication of the Hessian matrix H with the vector v.

            Args:
                v: Vector to multiply with Hessian (tensor)

            Returns:
                Hv_op: Hessian vector product op (tensor)
            """

            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape:
                    cost = self.cost_fun(self.y,self.model_fun(self.x,training=True))
                    cost_gradient = self.flatten(tape.gradient(target=cost, sources=self.params))
                    v = tf.transpose(tf.stop_gradient(v))
                    vprod = tf.tensordot(cost_gradient, v, axes = 0)
            Hv_op = self.flatten(tape2.gradient(target = vprod,sources= self.params))
            return Hv_op

        def get_H_op(self):
            """
            Implements a full Hessian estimator op by forming p Hessian vector
            products using HessianEstimator.get_Hv_op(v) for all v's in R^P

            Args:
                None

            Returns:
                H_op: Hessian matrix op (tensor)
            """
            H_op = tf.map_fn(self.get_Hv_op, tf.eye(self.P, self.P), dtype='float32')
            return H_op

        def get_G_op(self):
            """
            Implements a Hessian matrix OPG approximation op by a per-example
            cost Jacobian matrix product

            Args:
                None

            Returns:
                G_op: Hessian matrix OPG approximation op (tensor)
            """

            ex_params = [[tf.identity(_params) \
                          for _params in self.params] \
                         for ex in range(self.batch_size_G)]

            ex_X = tf.split(self.x, self.batch_size_G)
            ex_y = tf.split(self.y, self.batch_size_G)
            ex_yhat_logits = [self.model_fun(_X, _params) \
                              for _X, _params in zip(ex_X,
                                                     ex_params)]
            ex_cost = [self.cost_fun(_y, _yhat_logits, _params) \
                       for _y, _yhat_logits, _params in zip(ex_y,
                                                            ex_yhat_logits,
                                                            ex_params)]
            ex_grads = tf.stack([self.flatten(tf.gradients(ex_cost[ex],
                                                           ex_params[ex])) \
                                 for ex in range(self.batch_size_G)])
            G_op = tf.matmul(tf.transpose(ex_grads),
                             ex_grads) / self.batch_size_G
            return G_op
