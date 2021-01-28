from tests.test_helpers import *
from optimizers.ClassicMomentumOptimizer import *
from utils.training import *


# To succesfully perform tuning it is necessary to update the
# update_hyperparameters function and choose the right value of eta.
# If you choose a value that is too low the hyperparameter optimalization
# will never converge. On the other hand if eta is too big the process
# will diverge and probably terminate prematurely.

def tuning_example_reg():
    # To succesfully perform tuning it is necessary to update the
    # update_hyperparameters function and choose the right value of eta.
    # If you choose a value that is too low the hyperparameter optimalization
    # will never converge. On the other hand if eta is too big the process
    # will diverge and probably terminate prematurely.

    lr = 0.9
    decay = 0.5
    eta = 1000000000
    CM_optimizer = ClassicMomentumOptimizer(learning_rate=lr, decay=decay)
    x, y = create_Reg_Dataset()

    model = create_Single_Neuron_NN(CM_optimizer)
    model.save_weights('model.h5')

    CM_loss = train_CM(model, x, y, CM_optimizer, epochs=2)
    loss_value1 = loss(model, x.flatten(), y.flatten(), True)
    d_decay, d_lr, rev_loss = reverse_training(model, x, y, CM_optimizer.v_preciserep,
                                               params=CM_optimizer.var_preciserep,
                                               learning_rate=CM_optimizer.l_rate, decay=CM_optimizer.decay,
                                               epochs=2)
    training_with_hypergrad(d_lr, d_decay, lr, decay, x, y, 10, loss_value1, model, eta)


def tuning_example_reg2():
    lr = 0.2
    decay = 0.8
    eta = 2
    CM_optimizer = ClassicMomentumOptimizer(learning_rate=lr, decay=decay)
    x, y = create_Reg_Dataset2()

    model = create_Four_Neuron_NN(CM_optimizer)
    model.save_weights('model.h5')

    CM_loss = train_CM(model, x, y, CM_optimizer, epochs=2)
    loss_value1 = loss(model, x.flatten(), y.flatten(), True)
    d_decay, d_lr, rev_loss = reverse_training(model, x, y, CM_optimizer.v_preciserep,
                                               params=CM_optimizer.var_preciserep,
                                               learning_rate=CM_optimizer.l_rate, decay=CM_optimizer.decay,
                                               epochs=2)
    training_with_hypergrad(d_lr, d_decay, lr, decay, x, y, 100, loss_value1, model, eta)
