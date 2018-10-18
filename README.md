# pryme
A POC of a friendly API for optimization.

# Design

The interface takes cues from [keras]() and [pymc3]() design that makes building arbitrary models easy.
We also consider the design as an example of user driven development. Ideally we would like to express
our models fluently as follows

```python
from pryme import Model, RealVariable, Constraint, argmax, Bound


with Model() as model:
    x = RealVariable('x', lower_bound=45)
    y = RealVariable('y')
    
    Bound(5) <= y <= Bound(20)

    Constraint(50*x + 24*y) <= 2400
    Constraint(30*x + 33*y) <= 2100

    objective = argmax(x + y - 50)
    
model.solve()
```
```
{'objective_value': 1.25,
 'solution': {<tf.Variable 'x_2:0' shape=() dtype=float32_ref>: 45.0,
  <tf.Variable 'y_2:0' shape=() dtype=float32_ref>: 6.249999999999989}}
```

The framework should also provide a functional API, allowing users to programatically define
and solve problems

```python
model = Model()
x = RealVariable('x', lower_bound=45)
y = RealVariable('y', lower_bound=5, upper_bound=20)
model.add_variables([x, y])

# a constraint can be straight algebra on model variables
c1 = 50*x + 24*y
model.add_constraint(c1, less_equal=2400)

# or, you can encapsulate the constraint into a function
# and even add it with a decorator syntax
@model.add_constraint(less_equal=2100)
def c2():
    return 30*x + 33*y

# objective functions follow the same rules that constraints do
# just call model.add_objective() with an expression and tell
# the model how to optimize the problem
objective = model.add_objective(x + y - 50, type='maximization')

# or use a decorator
# @model.add_objective(type='maximization')
# def objective():
#     return x + y - 50
# model.solve()

model.solve()
```

Users should also be able to work with variables using vectorized operations.

```python
import numpy as np

import pryme


lower_bounds = np.array([45., 5.])
upper_bounds = np.array([None, 20.])

c1_coefs = np.array([50., 24.])
c2_coefs = np.array([30., 33.])
objective_coefs = np.array([1., 1.])


with Model() as model:
    x = RealVariable('x', shape=2, lower_bound=lower_bounds, upper_bound=upper_bounds)

    @model.add_constraint(less_equal=2400)
    def c1():
        return pryme.backend.dot(x, c1_coefs)

    @model.add_constraint(less_equal=2100)
    def c2():
        return pryme.backend.dot(x, c2_coefs)

    objective = argmax(pryme.backend.dot(x, objective_coefs) - 50)
    
model.solve()
```

Lastly the framework should be flexible allowing users to mix native operations
with the backend of choice (such as defining custom gradient calculations).

```python
import tensorflow as tf

h0 = 600.
s0 = (20_000 - 20*h0) / 170

def gradient(expression, wrt):
    grad = tf.gradients(expression, wrt)
    return grad / tf.norm(grad, ord=2)

with Model() as model:
    h = RealVariable('h', lower_bound=0)
    s = RealVariable('s', lower_bound=0)
    
    Constraint(20*h + 170*s) == 20_000
    
    objective = argmax(200*h**(2/3) * s**(1/3))
    
model.solve(x0=np.array([h0, s0]), gradient=gradient)
```
```
{'objective_value': 51854.8158311607,
 'solution': {'h:0': 666.6666664249711, 's:0': 39.21568630321231}}
```
