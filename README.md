# pryme
A POC of a friendly API for optimization.

# Design

The interface takes cues from [keras]() and [pymc3]() design that makes building arbitrary models easy.
We also consider the design as an example of user driven development. Ideally we would like to express
our models fluently as follows

```python
from pryme import LinearProgram, Var, Objective


# roughly taken from
# http://people.brunel.ac.uk/~mastjjb/jeb/or/morelp.html
with LinearProgram() as model:
    x = Var()
    y = Var()

    # bounds on variables
    x >= 45
    5 <= y <= 20
    
    # constraints
    50*x + 24*y <= 2400
    30*x + 33*y <= 2100
    
    objective = x + y -50
    
model.maximize(objective)
```
