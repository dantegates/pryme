_CONTEXT = []


def get_current_model():
    return _CONTEXT[-1]


def append_context(obj):
    _CONTEXT.append(obj)


def pop_context():
    return _CONTEXT.pop()