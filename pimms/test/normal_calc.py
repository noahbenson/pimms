# Usage example: implementing a normal-distribution object
import numpy as np
import pimms

# functions can now return the data itself instead of a dict
# if there is only one output, or, if there are more than one,
# they can return a tuple with the data in the same order
# as in the output values
@pimms.calc('variance')
def variance_from_stddev(standard_deviation):
    return standard_deviation ** 2

@pimms.calc('ci95', 'ci99')
def confidence_intervals(mean, standard_deviation):
    ci95 = tuple(q * standard_deviation + mean for q in [-1.96, 1.96])
    ci99 = tuple(q * standard_deviation + mean for q in [-2.58, 2.58])
    return (ci95, ci99)
    # or return {'ci95': ci95, 'ci99': ci99}

# calc functions that are decorated with None explicitly are
# always called whenever any of their dependent parameters are
# updated or when the new dictionary is created; this lets them
# act as checks or tests or requirements; they should not
# return anything
@pimms.calc(None)
def check_normal_distribution_inputs(standard_deviation):
    if standard_deviation <= 0:
        raise ValueError('standard_deviation must be positive')

# If the output name is not provided, it is, by default, the
# name of the function
@pimms.calc
def outer_normal_distribution_constant(standard_deviation):
    return 1.0 / (standard_deviation * np.sqrt(np.pi * 2))
@pimms.calc
def inner_normal_distribution_constant(standard_deviation):
    return -0.5 / standard_deviation

# Now we declare the calc plan: it's just the union of all these
# calc functions, given in any order:
normal_distribution = pimms.plan(
    variance=variance_from_stddev,
    cis=confidence_intervals,
    tests=check_normal_distribution_inputs,
    outer=outer_normal_distribution_constant,
    inner=inner_normal_distribution_constant)

# This function accepts a normal_distribution object and returns
# the probability density at a point
def pdf(nd, x):
    a = nd['outer_normal_distribution_constant']
    b = nd['inner_normal_distribution_constant']
    mu = nd['mean']
    return a * np.exp(b * (x - mu)**2)



