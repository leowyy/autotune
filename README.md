AutoTune
=========
How can we optimize the hyperparameters of a machine learning model efficiently?

Tuning a conv net on CIFAR10
-----
<img src="https://github.com/signapoop/autotune/blob/master/img/cifar_9hps.png" width="560">

Defining a problem instance
-----
1) Specify your hyperparameter domain in `self.initialise_domain()`, eg.
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

2) Specify your objective function in `self.eval_arm(params, n_resources)`, eg.
```python
def eval_arm(self, params, n_resources):
    model = generate_net(params)
    model.train(n_iter=n_resources)
    acc = model.test()
    return 1-acc
```

3) Test your problem with the following code snippet:
```python
problem = MyProblem()
params = problem.generate_arms(1)  # Draws a sample from the hyperparameter space
f_val = problem.eval_arm(params[0], 1)  # Evaluates the set of hyperparameters
```

References
-----
 - L. Li *et al* (2016), Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization

___Stay tuned for more updates...___
