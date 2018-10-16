import functools
from functools import partial

import numpy as np
import scipy.optimize
import tensorflow as tf


def run_expression(expression, variable_values):
    with tf.Session() as sess:
        for variable, value in variable_values.items():
            sess.run(variable.assign(value))
        if hasattr(expression, 'eval'):
            output = expression.eval()
        else:
            output = sess.run(expression)
    return output


# arithmetic on `Variable`s should return an "expression" - something
# that we can run with `run_expression` and get updated values.
class Variable(tf.Variable):
    pass


def _scipy_adapter(expression, variables):
    def wrapper(x):
        variable_values = {var: val for var, val in zip(variables, x)}
        return run_expression(expression, variable_values=variable_values)
    return wrapper


def minimize(objective, variables, constraints):
    solution = _minimize(objective, variables, constraints)
    objective_value = run_expression(objective, solution)
    return {'solution': solution, 'objective_value': objective_value}


def maximize(objective, variables, constraints):
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
    solution = _minimize(-1 * objective, variables, constraints)
    objective_value = run_expression(objective, solution)
    return {'solution': {var.name: val for var, val in solution.items()},
            'objective_value': objective_value}


def _minimize(objective, variables, constraints):
    l_bounds, u_bounds = _make_bounds(variables)
    tmp_result = scipy.optimize.minimize(
        _scipy_adapter(objective, variables=variables),
        x0=l_bounds,
        jac=_scipy_adapter(_gradient(objective, wrt=variables), variables=variables),
        bounds=scipy.optimize.Bounds(l_bounds, u_bounds),
        constraints=_make_constraints(variables, constraints),
        method='SLSQP')
    result = {var: x for var, x in zip(variables, tmp_result.x)}
    return result


def _make_bounds(variables):
    bounds = [(v.lower_bound, v.upper_bound) for v in variables]
    lower_bounds, upper_bounds = zip(*bounds)
    lower_bounds = [-np.inf if x is None else x for x in lower_bounds]
    upper_bounds = [np.inf if x is None else x for x in upper_bounds]
    return np.array(lower_bounds), np.array(upper_bounds)


def _make_constraints(variables, constraints):
    out = []
    for c in constraints:
        f = _scipy_adapter(c.expression, variables=variables)
        gradient_expression = _gradient(c.expression, wrt=variables)
        f_gradient = _scipy_adapter(gradient_expression, variables=variables)
        out.append(dict(type=c.type, fun=f, jac=f_gradient))
    return out


def _gradient(expression, wrt):
    return tf.gradients(expression, wrt)


def dot(x, y):
    return tf.tensordot(x, y, axes=1)
