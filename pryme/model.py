from functools import partial, singledispatch

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

    def add_constraint(self, constraint=None, *, less_equal=None, greater_equal=None, equal=None):
        if constraint is not None:
            if not isinstance(constraint, Constraint):
                user_experession = constraint() if callable(constraint) else constraint
                if less_equal is not None:
                    type = 'ineq'
                    expression = less_equal - user_experession
                elif greater_equal is not None:
                    type = 'ineq'
                    expression = user_experession - greater_equal
                elif equal is not None:
                    type = 'eq'
                    expression = user_experession
                else:
                    raise ValueError
                constraint = Constraint(expression=expression, type=type)
            self.constraints.append(constraint)
        else:
            return partial(self.add_constraint, less_equal=less_equal,
                           greater_equal=greater_equal)

    def minimize(self, objective):
        raise NotImplementedError

    def maximize(self, objective):
        raise NotImplementedError
        

class Model(BaseModel):
    def minimize(self, objective):
        if callable(objective):
            objective = objective()  # obtain an expression
        return backend.minimize(objective, self.variables, self.constraints)
    
    def maximize(self, objective):
        if callable(objective):
            objective = objective()  # obtain an expression
        return backend.maximize(objective, self.variables, self.constraints)


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

    def __le__(self, other):
        if isinstance(other, Bound):
            self.upper_bound = other.value
            return self
        return super().__le__(other)

    def __ge__(self, other):
        if isinstance(other, Bound):
            self.lower_bound = other.value
            return self
        return super().__ge__(other)


class Bound(ReprMixin):
    def __init__(self, value):
        self.value = value


class Constraint(ReprMixin):
    def __init__(self, value=None, *, expression=None, type=None):
        self.value = value
        self.expression = expression
        self.type = type

    def __eq__(self, other):
        self._register_with_current_model()
        self.expression = other
        self.type = 'eq'
        return other

    def __le__(self, other):
        self._register_with_current_model()
        self.expression = other - self.value
        self.type = 'ineq'
        return other

    def __ge__(self, other):
        self._register_with_current_model()
        self.expression = self.value - other
        self.type = 'ineq'
        return other

    def _register_with_current_model(self, **kwargs):
        try:
            current_model = context.get_current_model()
        except IndexError:
            pass
        else:
            current_model.add_constraint(self)
