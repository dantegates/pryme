# pryme
A POC of a friendly API for optimization.

# Design

The interface takes cues from [keras]() and [pymc3]() design that makes building arbitrary models easy.
We also consider the design as an example of user driven development. Ideally we would like to express
our models fluently as follows

```python
from pryme import Model, RealVariable, constraint

# roughly taken from
# http://people.brunel.ac.uk/~mastjjb/jeb/or/morelp.html
from pryme import Model, RealVariable


# roughly taken from
# http://people.brunel.ac.uk/~mastjjb/jeb/or/morelp.html
with Model() as model:
    x = RealVariable('x', lower_bound=45)
    y = RealVariable('y', lower_bound=5, upper_bound=20)
    
    @model.constraint(less_equal=2401)
    def c1():
        return 50*x + 24*y

    @model.constraint(less_equal=2100)
    def c2():
        return 30*x + 33*y

def objective():
    return x + y - 50

model.maximize(objective)
```
{<tf.Variable 'x:0' shape=() dtype=float32, numpy=45.07278>: 45.072779832561416,
 <tf.Variable 'y:0' shape=() dtype=float32, numpy=6.387564>: 6.387564146508517,
 'objective_value': 1.4603462}
```
