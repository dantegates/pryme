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


def _get_shape(variable):
    return tuple(dim.value for dim in variable.get_shape().dims)


def _map_x_to_variable_indices(variables):
    variable_indices = []
    i = 0
    for var in variables:
        try:
            var_len = _get_shape(var)[0]
        except IndexError:
            var_len = 1
            variable_indices.append((var, i))
        else:
            variable_indices.append((var,  slice(i, i+var_len)))
        i += var_len
    return variable_indices


def _scipy_adapter(expression, variables, name=None):
    variable_indices = _map_x_to_variable_indices(variables)
    def wrapper(x):
        variable_values = {var: x[idx] for var, idx in variable_indices}
        output = run_expression(expression, variable_values=variable_values)
        return output
    return wrapper


def minimize(objective, variables, constraints, **kwargs):
    solution = _minimize(objective, variables, constraints, **kwargs)
    objective_value = run_expression(objective, solution)
    return {'solution': solution, 'objective_value': objective_value}


def maximize(objective, variables, constraints, **kwargs):
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
    solution = _minimize(-1 * objective, variables, constraints, **kwargs)
    objective_value = run_expression(objective, solution)
    return {'solution': {var.name: val for var, val in solution.items()},
            'objective_value': objective_value}


def _minimize(objective, variables, constraints, x0=None, gradient=None, **kwargs):
    l_bounds, u_bounds = _make_bounds(variables)
    gradient = _gradient if gradient is None else gradient
    result = scipy.optimize.minimize(
        _scipy_adapter(objective, variables=variables, name='objective'),
        x0=l_bounds if x0 is None else x0,
        jac=_scipy_adapter(gradient(objective, wrt=variables), variables=variables, name='objective grad'),
        bounds=scipy.optimize.Bounds(l_bounds, u_bounds),
        constraints=_make_constraints(variables, constraints, gradient),
        method='SLSQP',
        options=kwargs if kwargs else {})
    variable_indices = _map_x_to_variable_indices(variables)
    return {var: result.x[idx] for var, idx in variable_indices}


def _make_bounds(variables):
    lower_bounds = []
    upper_bounds = []
    for var in variables:
        try:
            lower_bounds.extend(var.lower_bound)
            upper_bounds.extend(var.upper_bound)
        except TypeError:
            lower_bounds.append(var.lower_bound)
            upper_bounds.append(var.upper_bound)
    lower_bounds = [-np.inf if x is None else x for x in lower_bounds]
    upper_bounds = [np.inf if x is None else x for x in upper_bounds]
    return np.array(lower_bounds), np.array(upper_bounds)


def _make_constraints(variables, constraints, gradient):
    out = []
    for c in constraints:
        f = _scipy_adapter(c.expression, variables=variables, name='constraint')
        gradient_expression = gradient(c.expression, wrt=variables)
        f_gradient = _scipy_adapter(gradient_expression, variables=variables, name='constraint_grad')
        out.append(dict(type=c.type, fun=f, jac=f_gradient))
    return out


def _gradient(expression, wrt):
    return tf.gradients(expression, wrt)


def dot(x, y):
    return tf.tensordot(x, y, axes=1)
