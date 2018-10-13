import numpy as np
import scipy.optimize
import tensorflow as tf

# tf.enable_eager_execution()

tfe = tf.contrib.eager # shorthand for some symbols


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
        self.bounds = {}


    def __enter__(self):
        _CONTEXT.append(self)
        return self

    def __exit__(self, *exc):
        _CONTEXT.pop()
        return False
    
    def add_variable(self, variable):
        self.variables.append(variable)
        
    def add_bound(self, var, bound, kind):
        if not var.name in self.bounds:
            self.bounds[var.name]= [None, None]
        if kind == 'lower':
            self.bounds[var.name][0] = bound
        else:
            self.bounds[var.name][1] = bound
        
    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def minimize(self, objective):
        raise NotImplementedError

    def maximize(self, objective):
        raise NotImplementedError


class Model(BaseModel):
    def minimize(self, objective_fn):
        pass

    def _minimize(self, objective_fn, bounds, constraints):
        l_bounds, u_bounds = self._make_bounds(bounds)
        return scipy.optimize.minimize(
            lambda x: objective_fn(*x),
            x0=l_bounds,
            jac=lambda x: gradient(objective_fn)(*x),
            bounds=scipy.optimize.Bounds(l_bounds, u_bounds),
            constraints=self._make_constraints(constraints),
            method='SLSQP')
    
    def _make_bounds(self, bounds):
        var_names = [v.name for v in self.variables]
        bounds = sorted(bounds.items(), key=lambda x: var_names.index(x[0]))
        bounds = [v for k, v in bounds]
        lower_bounds, upper_bounds = zip(*bounds)
        lower_bounds = [-np.inf if x is None else x for x in lower_bounds]
        upper_bounds = [np.inf if x is None else x for x in upper_bounds]
        return np.array(lower_bounds), np.array(upper_bounds)
    
    @staticmethod
    def _make_constraints(constraints):
        return [
            dict(type='ineq', fun=lambda x: c(*x), jac=lambda x: gradient(c)(*x))
            for c in constraints
        ]

    def maximize(self, objective):
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
        return self._minimize(
            lambda *args: -1 * objective(*args),
            self.bounds,
            self.constraints)
#             [lambda *args: -1 * c(*args) for c in self.constraints])


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
        

    def __le__(self, other):
        current_model = _get_current_model()
        current_model.add_bound(self, other, 'upper')
        return self

    def __ge__(self, other):
        current_model = _get_current_model()
        current_model.add_bound(self, other, 'lower')
        return self
        
        
class RealVariable(BaseVariable):
    pass


def gradient(f):
    def inner(*args):
        grad = tfe.gradients_function(f)(*args)
        return np.array([x.numpy() for x in tfe.gradients_function(f)(*args)])
    return inner
    

def constraint(f):
    current_model = _get_current_model()
    current_model.add_constraint(f)
    return f
