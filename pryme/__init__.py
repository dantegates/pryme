import numbers


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
    def minimize(self, objective, bounds, constraints):
        raise NotImplementedError

    def maximize(self, objective, bounds, constraints):
        """Maximize the objective function.

        Args:
            objective (`numpy.array`): Numpy array of length `n` representing the
                linear coefficients for the model's variables.
            bounds (`list` of `tuple`s or `None): List of length `n` of `tuple`s 
                representing the bounds on the model's variables or None if the
                variable is unbounded. If a `tuple` should have two items either
                an instance of `Bound` or None. The first represents a lower
                bound and the second represents an upper bound.
            constraints (`list` of `numpy.array`s): Each numpy array should be
                of length `n` and each element represents the coefficient of
                the corresponding variable in the array.
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
