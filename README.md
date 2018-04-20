### AutoTune: How can we optimize the hyperparameters of a machine learning model efficiently?

__Tuning a convolutional net on CIFAR10, 9 hyperparam__
<img src="https://github.com/signapoop/autotune/blob/master/img/cifar_9hps.png" width="600">

Defining a problem instance
-----
1) Specify the domain of hyperparameters in `self.initialise_domain()`, eg:
```python
params = {
            'learning_rate': Param('learning_rate', -6, 0, distrib='uniform', scale='log', logbase=10),
            'weight_decay': Param('weight_decay', -6, -1, distrib='uniform', scale='log', logbase=10),
            'momentum': Param('momentum', 0.3, 0.999, distrib='uniform', scale='linear'),
            'batch_size': Param('batch_size', 20, 2000, distrib='uniform', scale='linear', interval=1),
        }
```
2) Specify the objective function in `self.initialise_objective_function(x)`

3) Test the problem with the following code snippet:
```python
problem = MyProblem()
arm = problem.generate_random_arm()
f_val = problem.eval_arm(arm)
```

__Stay tuned for more updates...__
