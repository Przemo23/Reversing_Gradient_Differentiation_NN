# Reversing_Gradient_Differentiation_NN

## About
This project is the Tensorflow implementation of the gradient based hyperparameter optimalization using
the reversibility of the Classic Momentum algorithm, which was presented by D.Maclaurin, D.Duvenaud and R.Adams
in https://arxiv.org/abs/1502.03492. The goal is to access the learning rate and decay gradients of the learning process
of a neural network.

## Structure

The description of the files in this project:
1. *optimizers* directory contains the implementations of the optimizers based on the Classic Momentum algorithm and its reverse form,
2. *pyhessian* directory contains the module, which calculates the hessian used in the calculation of the decay gradient,
3. *tests* directory is self-explanatory,
4. *utils* directory contains the *preciseRep* class which is responsible for fixed-point arithmetic in this project. The directory also contains the *training.py* file which provides useful functions for the set up of the learning process of an NN.,
5. *tuning_example* file includes a couple examples of hyperparameter tuning,
6. *main.py* file runs the tests and examples.

## Setup

### Python and pip
Firstly make sure you have Python and pip installed on your machine.\
You can check it by typing into the console:

```
> python --version
Python 3.8.x
> pip --version
pip x.x.x from ... (python 3.8)
```

### Cloning the repo
Clone the repository to a directory of your choice
```
git clone https://github.com/Przemo23/Reversing_Gradient_Differentiation_NN.git
```

### Creating an virtual enviroment
Next step is to create an virtualenv. If you don't have virtualenv just run:
```
pip install virtualenv
```
Then you have to create the venv
```
> python -m venv rgd-cm-env
```
To activate it run:
Windows:
```
> .\rgd-cm-env\Scripts\activate
```
Linux:
```
> source rgd-cm-env/bin/activate
```

### Installing the necessary libraries:
```
(rgd-cm-env) > pip install -r requirements.txt
```

### Running the tests and tuning examples:
```
(rgd-cm-env) > python main.py
```


## Disclaimer
The pyhessian and preciseRep classes are adaptations of the solutions created in https://github.com/gknilsen/pyhessian 
and https://github.com/HIPS/hypergrad respectively. 


## Sources:
1. Maclaurin, Duvenaud, Adams, Gradient-based Hyperparameter Optimization through Reversible Learning
https://arxiv.org/abs/1502.03492
2. Ian Goodfellow and Yoshua Bengio and Aaron Courville Deep Learning. MIT Press, 2016, http://www.deeplearningbook.org
3. Barak A. Pearlmutter Fast Exact Multiplication by the Hessian Neural Computation, 1993,
http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf
4. Nielsen, Antonella, Munthe-Kaas, Skaug, Brun, Efficient Computation of Hessian Matrices in TensorFlow
https://arxiv.org/abs/1905.05559
5. Benoit Descamps, Custom Optimizer in TensorFlow, https://www.kdnuggets.com/2018/01/custom-optimizer-tensorflow.html
6. Tensorflow, https://www.tensorflow.org/
