import numpy as np
import scipy.optimize


_CONTEXT = []


def _public_attrs(obj):
    attrs = ((attr, getattr(obj, attr))
             for attr in dir(obj)
             if not attr.startswith('_')
             and not callable(getattr(obj, attr)))
    return sorted(attrs)


class ReprMixin:
    def __repr__(self):
        name = getattr(self, '__name__', self.__class__.__name__)
        attrs = ', '.join(f'{attr}={val!r}' for attr, val in _public_attrs(self))
        return f'{name}({attrs})'


def _get_current_model():
    return _CONTEXT[-1]


class BaseModel(ReprMixin):
    def __init__(self):
        self.variables = []
        self.constraints = []
        self.bounds = []


    def __enter__(self):
        _CONTEXT.append(self)
        return self

    def __exit__(self, *exc):
        _CONTEXT.pop()
        return False
    
    def add_variable(self, variable):
        self.variables.append(variable)
        
    def add_bound(self, bound):
        self.bounds.append(bound)

    def minimize(self, objective):
        raise NotImplementedError

    def maximize(self, objective):
        raise NotImplementedError


class LinearProgram(BaseModel):
    def minimize(self, objective):
        pass

    def _minimize(self, objective, bounds, constraints):
        def objective_fn(x):
            return np.dot(objective[:-1], x) - objective[-1]
        def objective_gradient_fn(x):
            return objective[:-1]
        lower_bounds, upper_bounds = zip(*bounds)
        lower_bounds = [-np.inf if x is None else x for x in lower_bounds]
        upper_bounds = [np.inf if x is None else x for x in upper_bounds]
        bounds = scipy.optimize.Bounds(np.array(lower_bounds), np.array(upper_bounds))
        x0 = np.array(lower_bounds)
        constraints = [
            dict(type='ineq', fun=lambda x: np.dot(c[:-1], x) - c[-1], jac=lambda x: c[:-1])
            for c in constraints
        ]
        return scipy.optimize.minimize(
            objective_fn,
            x0=x0,
            jac=objective_gradient_fn,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP')
        

    def maximize(self, objective, bounds, constraints):
        """Maximize the objective function.

        Args:
            objective (`numpy.array`): Numpy array of length `n+1` representing the
                linear coefficients for the model's variables and any constants
                added or subtracted to the linear combination.
            bounds (`list` of `tuple`s or `None): List of length `n` of `tuple`s 
                representing the bounds on the model's variables or None if the
                variable is unbounded. If a `tuple` should have two items. The
                first is a scalar or None representing the lower bound or lack
                thereof. The second is similar and represents the upper bound.

            constraints (`list` of `numpy.array`s.): `numpy.array` 
                of length `n+1` and each element represents the coefficient of
                the corresponding variable in the array and the last corresponding
                to the right hand side of the equation.
        """
        return self.minimize(-objective, bounds, [-c for c in constraints])


class Bound:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        current_model = _get_current_model()
        current_model.add_bound(instance)
        return instance

    def __init__(self, left, right, comparison):
        self.left = left
        self.right = right
        self.comparison = comparison
        
    def __repr__(self):
        return f'{self.left} {self.comparison} {self.right}'


class BaseVariable:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        current_model = _get_current_model()
        current_model.add_variable(instance)
        return instance
    
    def __init__(self, name):
        self.name = name
        
    def __repr__(self):
        return f'{self.__class__.__name__}({self.name!r})'
        
    def __lt__(self, other):
        Bound(self, other, '<')
        return self
    
    def __le__(self, other):
        Bound(self, other, '<=')
        return self

    def __gt__(self, other):
        Bound(self, other, '>')
        return self
    
    def __ge__(self, other):
        Bound(self, other, '>=')
        return self
    
    def __eq__(self, other):
        raise NotImplementedError('not implementing this yet')
        
    def __ne__(self, other):
        raise NotImplementedError('not implementing this yet')
        
        
class RealVariable(BaseVariable):
    pass
