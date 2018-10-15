from functools import partial

import numpy as np

from . import backend
from . import context
from .utils import ReprMixin


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

    def add_variables(self, variables):
        for variable in variables:
            self.add_variable(variable)
    
    def add_variable(self, variable):
        self.variables.append(variable)

    def constraint(self, fn=None, **kwargs):
        if fn is None:
            return partial(self.constraint, **kwargs)
        constraint = backend.Constraint(fn, **kwargs)
        self.constraints.append(constraint)
        return fn

    add_constraint = constraint

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
            var.assign(val)


class BaseVariable(backend.Variable):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        try:
            current_model = context.get_current_model()
        except IndexError:
            pass
        else:
            current_model.add_variable(instance)
        return instance

    def __init__(self, name, *, lower_bound=None, upper_bound=None, **kwargs):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        initial_val = self._initial_val()
        super().__init__(initial_val, name=name, **kwargs)

    def _initial_val(self):
        return NotImplementedError


class RealVariable(BaseVariable):
    def __init__(self, *args, shape=None, lower_bound=-np.inf, upper_bound=np.inf, **kwargs):
        # name mangling
        self.__shape = shape
        super().__init__(*args, lower_bound=lower_bound, upper_bound=upper_bound, **kwargs)

    def _initial_val(self):
        return 0.0
