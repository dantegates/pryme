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
with Model() as model:
    x = RealVariable('x')
    y = RealVariable('y')

    # bounds on variables
    x >= 45
    5 <= y <= 20
    
    # constraints
    @constraint
    def c1(x, y):
        return 2400 - 50*x - 24*y
        
    @constraint
    def c2(x, y):
        return 2100 - 30*x - 33*y
    
def objective(x, y):
    return x + y - 50

model.maximize(objective)
```
```
{RealVariable('x'): 45.0618395369519, RealVariable('y'): 6.47264727553161}
```
