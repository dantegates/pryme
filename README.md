# pryme
A POC of a friendly API for optimization.

# Design

The interface takes cues from [keras]() and [pymc3]() design that makes building arbitrary models easy.
We also consider the design as an example of user driven development. Ideally we would like to express
our models fluently as follows

```python
from pryme import Model, RealVariable, Bound as B, Constraint as C


with Model() as model:
    x = RealVariable('x')
    y = RealVariable('y')

    x >= B(45)
    B(5) <= y <= B(20)

    C(2400) >= 50*x + 24*y
    C(2100) >= 30*x + 33*y

    objective = x + y - 50
    
model.maximize(objective)
    
```
{'objective_value': 1.25,
 'solution': {<tf.Variable 'x_2:0' shape=() dtype=float32_ref>: 45.0,
  <tf.Variable 'y_2:0' shape=() dtype=float32_ref>: 6.249999999999989}}
```
