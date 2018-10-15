_CONTEXT = []


def _get_current_model():
    return _CONTEXT[-1]


class BaseModel:
    def __init__(self):
        self.variables = []
        self.constraints = []
        self.bounds = []

    def __enter__(self):
        _CONTEXT.append(self)

    def __exit__(self, *exc):
        _CONTEXT.pop()
        return False


class LinearProgram(BaseModel):
    pass


class BaseVariable:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(*args, **kwargs)
        current_model = _get_current_model()
        current_model.add_var(instance)
        return instance


class RealVariable(BaseVariable):
    pass
