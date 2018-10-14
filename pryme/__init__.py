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


def _func_wrapper(f, model):
    def wrapper(x):
        model.update_vars(x)
        return f()
    return wrapper
        

class Model(BaseModel):
    def minimize(self, objective_fn):
        pass

    def _minimize(self, objective_fn, bounds, constraints):
        wrt = [var.val for var in model.variables]
        l_bounds, u_bounds = self._make_bounds(bounds)
        return scipy.optimize.minimize(
            _func_wrapper(objective_fn, model=model),
            x0=l_bounds,
            jac=_func_wrapper(lambda: gradient(objective_fn, wrt=wrt), model=model),
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
        wrt = [var.val for var in model.variables]
        return [
            dict(type='ineq', fun=_func_wrapper(c, model=model),
                 jac=_func_wrapper(lambda: gradient(c, wrt=wrt), model=model))
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
    
    def update_vars(self, x):
        for val, var in zip(x, self.variables):
            var.update(val)


class BaseVariable:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        current_model = _get_current_model()
        current_model.add_variable(instance)
        return instance
    
    def __init__(self, name):
        self.name = name
        self.val = tfe.Variable(0.0, name=self.name)
        
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
    
    def __mul__(self, other):
        return self.val * getattr(other, 'val', other)
    
    def __rmul__(self, other):
        return self.val * getattr(other, 'val', other)
    
    def __add__(self, other):
        return self.val + getattr(other, 'val', other)
    
    def __sub__(self, other):
        return self.val - getattr(other, 'val', other)
    
    def __radd__(self, other):
        return self.val + getattr(other, 'val', other)
    
    def __rsub__(self, other):
        return other - getattr(other, 'val', other)
    
    def update(self, value):
        self.val.assign(value)
        
        
class RealVariable(BaseVariable):
    pass


def gradient(f, wrt):
    with tf.GradientTape(persistent=True) as t:
        t.watch(wrt)
        f_eval = f()
    df = t.gradient(f_eval, wrt)
    return np.array([grad.numpy() for grad in df])
    

def constraint(f):
    current_model = _get_current_model()
    current_model.add_constraint(f)
    return f
