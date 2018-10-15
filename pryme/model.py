from functools import partial

import numpy as np

from . import backend
from . import context


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


class BaseModel(ReprMixin):
    def __init__(self):
        self.variables = []
        self.constraints = []

    def __enter__(self):
        context.append_context(self)
        return self

    def __exit__(self, *exc):
        context.pop_context()
        return False
    
    def add_variable(self, variable):
        self.variables.append(variable)

    def constraint(self, fn=None, **kwargs):
        if fn is None:
            return partial(self.constraint, **kwargs)
        constraint = backend.constraint(fn, **kwargs)
        self.constraints.append(constraint)
        return fn
        
    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def minimize(self, objective):
        raise NotImplementedError

    def maximize(self, objective):
        raise NotImplementedError
        

class Model(BaseModel):
    def minimize(self, objective_fn):
        return backend.minimize(self, objective_fn, self.variables, self.constraints)
    
    def maximize(self, objective_fn):
        return backend.maximize(self, objective_fn, self.variables, self.constraints)
    
    def update_vars(self, x):
        for val, var in zip(x, self.variables):
            var.update(val)


class BaseVariable(ReprMixin):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        current_model = context.get_current_model()
        current_model.add_variable(instance)
        return instance

    def __init__(self, name, lower_bound=-np.inf, upper_bound=np.inf):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._val = backend.Variable(0.0, name=self.name)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name!r})'

    def __le__(self, other):
        current_model = context.get_current_model()
        current_model.add_bound(self, other, 'upper')
        return self

    def __ge__(self, other):
        current_model = context.get_current_model()
        current_model.add_bound(self, other, 'lower')
        return self
    
    def __mul__(self, other):
        return self._val * getattr(other, '_val', other)
    
    def __rmul__(self, other):
        return self._val * getattr(other, '_val', other)
    
    def __add__(self, other):
        return self._val + getattr(other, '_val', other)
    
    def __sub__(self, other):
        return self._val - getattr(other, '_val', other)
    
    def __radd__(self, other):
        return self._val + getattr(other, '_val', other)
    
    def __rsub__(self, other):
        return other - getattr(other, '_val', other)

    def __pow__(self, other):
        return self._val ** other
    
    def update(self, value):
        self._val.assign(value)
        
        
class RealVariable(BaseVariable):
    pass
