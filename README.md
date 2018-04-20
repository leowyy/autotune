AutoTune
=========
How can we optimize the hyperparameters of a machine learning model efficiently?

Tuning a convolutional net on CIFAR10, 9 hyperparam
-----
<img src="https://github.com/signapoop/autotune/blob/master/img/cifar_9hps.png" width="560">

Defining a problem instance
-----
1) Specify your hyperparameter domain in `self.initialise_domain()`
```python
def initialise_domain(self):
    params = {
        'learning_rate': Param('learning_rate', -6, 0, distrib='uniform', scale='log', logbase=10),
        'weight_decay': Param('weight_decay', -6, -1, distrib='uniform', scale='log', logbase=10),
        'momentum': Param('momentum', 0.3, 0.9, distrib='uniform', scale='linear'),
        'batch_size': Param('batch_size', 20, 2000, distrib='uniform', scale='linear', interval=1),
    }
    return params
```

2) Specify your objective function in `self.initialise_objective_function(x)`
```python
def initialise_objective_function(self, params):
    model = generate_net(params)
    model.train()
    acc = model.test()
    return 1-acc
```

3) Test your problem with the following code snippet:
```python
problem = MyProblem()
params = problem.generate_random_arm()  # Draws a sample from the hyperparameter space
f_val = problem.eval_arm(params)  # Evaluates the set of hyperparameters
```

___Stay tuned for more updates...___
