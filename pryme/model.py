from functools import partial, singledispatch

import numpy as np

from . import backend
from .utils import ReprMixin


_CONTEXT = []


def get_current_model():
    return _CONTEXT[-1]


def append_context(obj):
    _CONTEXT.append(obj)


def pop_context():
    return _CONTEXT.pop()


class BaseModel(ReprMixin):
    def __init__(self):
        self.variables = []
        self.constraints = []
        self.objective = None
        self.flavor = None

    def __enter__(self):
        append_context(self)
        return self

    def __exit__(self, *exc):
        pop_context()
        return False

    def add_variables(self, variables):
        for variable in variables:
            self.add_variable(variable)
    
    def add_variable(self, variable):
        self.variables.append(variable)

    def add_constraint(self, constraint=None, *, less_equal=None, greater_equal=None, equal=None):
        if constraint is None:
            return partial(self.add_constraint, less_equal=less_equal,
                           greater_equal=greater_equal, equal=equal)
        elif not isinstance(constraint, Constraint):
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
            c = Constraint(expression=expression, type=type)
        else:
            c = constraint
        self.constraints.append(c)
        return constraint

    def add_objective(self, objective=None, *, type):
        if objective is None:
            return partial(self.add_objective, type=type)
        if callable(objective):
            objective = objective()  # obtain an expression
        self.objective = objective
        self.type = type
        return objective

    def solve(self):
        raise NotImplementedError
        

class Model(BaseModel):
    def solve(self, **kwargs):
        solver = backend.minimize if self.type == 'minimization' else backend.maximize
        return solver(self.objective, self.variables, self.constraints, **kwargs)


class BaseVariable(backend.Variable):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        try:
            current_model = get_current_model()
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
    def __init__(self, *args, shape=None, lower_bound=None, upper_bound=None, **kwargs):
        # name mangling
        self.__shape = shape
        lower_bound, upper_bound = self._default_bounds(lower_bound, upper_bound)
        super().__init__(*args, lower_bound=lower_bound, upper_bound=upper_bound, **kwargs)

    def _initial_val(self):
        if self.__shape is not None:
            return np.zeros(self.__shape, dtype=np.float64)
        return np.float64(0.0)

    def _default_bounds(self, lower_bound, upper_bound):
        if lower_bound is None:
            if self.__shape is None:
                lower_bound = -np.inf
            else:
                lower_bound = np.repeat(-np.inf, self.__shape)
        if upper_bound is None:
            if self.__shape is None:
                upper_bound = np.inf
            else:
                upper_bound = np.repeat(np.inf, self.__shape)
        return lower_bound, upper_bound

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
    def __init__(self, expression, *, type=None):
        self.expression = expression
        self.type = type

    def __eq__(self, other):
        self._register_with_current_model()
        self.type = 'eq'
        return other

    def __le__(self, other):
        self._register_with_current_model()
        self.expression = other - self.expression
        self.type = 'ineq'
        return other

    def __ge__(self, other):
        self._register_with_current_model()
        self.expression = self.expression - other
        self.type = 'ineq'
        return other

    def _register_with_current_model(self, **kwargs):
        try:
            current_model = get_current_model()
        except IndexError:
            pass
        else:
            current_model.add_constraint(self)


def argmin(objective):
    model = get_current_model()
    model.add_objective(objective, type='minimization')
    return objective


def argmax(objective):
    model = get_current_model()
    model.add_objective(objective, type='maximization')
    return objective
