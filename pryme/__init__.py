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


class LinearProgram(BaseModel):
    pass


class BaseVariable(ReprMixin):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        current_model = _get_current_model()
        current_model.add_variable(instance)
        return instance
    
    def __init__(self, name):
        self.name = name
        self.lower_bound = None
        self.upper_bound = None
        
    def __lt__(self, other):
        self.upper_bound = (other, 'lt')
        return self
    
    def __le__(self, other):
        self.upper_bound = (other, 'le')
        return self

    def __gt__(self, other):
        self.lower_bound = (other, 'gt')
        return self
    
    def __ge__(self, other):
        self.lower_bound = (other, 'ge')
        return self
    
    def __eq__(self, other):
        raise NotImplementedError('not implementing this yet')
        
    def __ne__(self, other):
        raise NotImplementedError('not implementing this yet')
        
        
class RealVariable(BaseVariable):
    pass
