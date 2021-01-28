from tests.test_helpers import *
from optimizers.ClassicMomentumOptimizer import *
from utils.training import *


def tuning_example():
    # To succesfully perform tuning it is necessary to update the
    # update_hyperparameters function and choose the right value of eta.
    # If you choose a value that is too low the hyperparameter optimalization
    # will never converge. On the other hand if eta is too big the process
    # will diverge and probably terminate prematurely.


    lr = 0.9
    decay = 0.5
    CM_optimizer = ClassicMomentumOptimizer(learning_rate=lr, decay=decay)
    x, y = create_Reg_Dataset()

    model = create_Single_Neuron_NN(CM_optimizer)
    init_weights = model.get_weights()
    model.save_weights('model.h5')

    CM_loss = train_CM(model, x, y, CM_optimizer, epochs=2)
    loss_value1 = loss(model, x.flatten(), y.flatten(), True)
    d_decay, d_lr, rev_loss = reverse_training(model, x, y, CM_optimizer.v_preciserep,
                                               params=CM_optimizer.var_preciserep,
                                               learning_rate=CM_optimizer.l_rate, decay=CM_optimizer.decay,
                                               epochs=2)
    reversed_weights = model.get_weights()
    # vars = CM_loss + rev_loss
    # w1 = [abs(var[0][0] - init_weights[0][0]) for var in vars]
    # w2 = [abs(var[1] - init_weights[1]) for var in vars]
    #
    # plt.plot(range(128), w1,label="w")
    # plt.plot(range(128), w2,label="bias")
    #
    # plt.legend()
    #
    # plt.show()

    training_with_hypergrad( d_lr, d_decay, lr, decay, x, y, 10, loss_value1, model)

