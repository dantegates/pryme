import functools

import numpy as np
import scipy.optimize
import tensorflow as tf

from .utils import ReprMixin


tf.enable_eager_execution()

tfe = tf.contrib.eager # shorthand for some symbols


class Variable(tfe.Variable):
    pass


class Constraint(ReprMixin):
    def __init__(self, fn, less_equal=None, greater_equal=None):
        self.fn = fn.__name__
        if less_equal is not None:
            self.less_equal = less_equal
            def c():
                return less_equal - fn()
        elif greater_equal is not None:
            self.greater_equal = greater_equal
            def c():
                return fn() - greater_equal

        self._constraint = c

    def __call__(self):
        return self._constraint()


def _scipy_adapter(f, model):
    def wrapper(x):
        model.update_vars(x)
        return f()
    return wrapper


def minimize(model, objective_fn, variables, constraints):
    result = _minimize(model, objective_fn, variables, constraints)
    result['objective_value'] = objective_fn().numpy()
    return result


def maximize(model, objective_fn, variables, constraints):
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
    result = _minimize(model, _negate(objective_fn), variables, constraints)
    result['objective_value'] = objective_fn().numpy()
    return result


def _minimize(model, objective_fn, variables, constraints):
    l_bounds, u_bounds = _make_bounds(variables)
    tmp_result = scipy.optimize.minimize(
        _scipy_adapter(objective_fn, model=model),
        x0=l_bounds,
        jac=_scipy_adapter(lambda: _gradient(objective_fn, wrt=variables), model=model),
        bounds=scipy.optimize.Bounds(l_bounds, u_bounds),
        constraints=_make_constraints(model, variables, constraints),
        method='SLSQP')
    result = {var: x for var, x in zip(variables, tmp_result.x)}
    return result


def _make_bounds(variables):
    bounds = [(v.lower_bound, v.upper_bound) for v in variables]
    lower_bounds, upper_bounds = zip(*bounds)
    lower_bounds = [-np.inf if x is None else x for x in lower_bounds]
    upper_bounds = [np.inf if x is None else x for x in upper_bounds]
    return np.array(lower_bounds), np.array(upper_bounds)


def _make_constraints(model, variables, constraints):
    return [
        dict(type='ineq', fun=_scipy_adapter(c, model=model),
             jac=_scipy_adapter(lambda: _gradient(c, wrt=variables), model=model))
        for c in constraints
    ]


def _gradient(f, wrt):
    with tf.GradientTape(persistent=True) as t:
        t.watch(wrt)
        f_eval = f()
    df = t.gradient(f_eval, wrt)
    return np.array([grad.numpy() for grad in df])


def _negate(f):
    @functools.wraps(f)
    def wrapper(*args):
        return -1 * f(*args)
    return wrapper
