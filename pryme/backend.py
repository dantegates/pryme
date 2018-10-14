import functools

import numpy as np
import scipy.optimize
import tensorflow as tf

from . import context


tf.enable_eager_execution()

tfe = tf.contrib.eager # shorthand for some symbols


Variable = tfe.Variable


def _scipy_adapter(f, model):
    def wrapper(x):
        model.update_vars(x)
        return f()
    return wrapper


def minimize(model, objective_fn, variables, bounds, constraints):
    wrt = [var.val for var in variables]
    l_bounds, u_bounds = _make_bounds(variables, bounds)
    result = scipy.optimize.minimize(
        _scipy_adapter(objective_fn, model=model),
        x0=l_bounds,
        jac=_scipy_adapter(lambda: gradient(objective_fn, wrt=wrt), model=model),
        bounds=scipy.optimize.Bounds(l_bounds, u_bounds),
        constraints=_make_constraints(model, variables, constraints),
        method='SLSQP')
    return {var: x for var, x in zip(variables, result.x)}


def maximize(model, objective_fn, variables, bounds, constraints):
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
    return minimize(
        model,
        negate(objective_fn),
        variables,
        bounds,
        constraints)

def _make_bounds(variables, bounds):
    var_names = [v.name for v in variables]
    bounds = sorted(bounds.items(), key=lambda x: var_names.index(x[0]))
    bounds = [v for k, v in bounds]
    lower_bounds, upper_bounds = zip(*bounds)
    lower_bounds = [-np.inf if x is None else x for x in lower_bounds]
    upper_bounds = [np.inf if x is None else x for x in upper_bounds]
    return np.array(lower_bounds), np.array(upper_bounds)


def _make_constraints(model, variables, constraints):
    wrt = [var.val for var in variables]
    return [
        dict(type='ineq', fun=_scipy_adapter(c, model=model),
             jac=_scipy_adapter(lambda: gradient(c, wrt=wrt), model=model))
        for c in constraints
    ]


def gradient(f, wrt):
    with tf.GradientTape(persistent=True) as t:
        t.watch(wrt)
        f_eval = f()
    df = t.gradient(f_eval, wrt)
    return np.array([grad.numpy() for grad in df])
    

def constraint(f):
    current_model = context.get_current_model()
    current_model.add_constraint(f)
    return f


def negate(f):
    @functools.wraps(f)
    def wrapper(*args):
        return -1 * f(*args)
    return wrapper