import pickle
import inspect
import re
import itertools
import numpy as np
from scipy import optimize
from default_parameters import el
# from TEC import physical_constants
physical_constants = {"boltzmann" : 1.3806488e-23, \
                      "permittivity0" : 8.85418781762e-12, \
                      "electron_charge" : 1.602176565e-19, \
                      "electron_mass" : 9.1093897e-31, \
                      "sigma0" : 5.67050e-8}

"""
Library to aid in generating standard data for tec numerical testing.
"""

def round_exponential(val):
    """
    Returns an integer [1-9] times a power of 10.

    Example:

    >>> round_exponential(5.75e-8)
    4.9999999999999998e-08
    >>> # What are you going to do? Its floating point math.
    >>> round_exponential(3.2804510630249406e12)
    3000000000000.0
    >>> round_exponential(5e2)
    500.0
    """
    expt = np.floor(np.log10(abs(val)))
    mant = val / 10**expt
    if val > 0:
        return np.floor(mant) * 10**expt
    else:
        return np.ceil(mant) * 10**expt

def gen_combos(params):
    """
    Generate all combos of parameters in a dict of lists.

    Returns a list of dicts of all combinations, the keys of each dict matching the keys of the `params` argument.

    :param dict params: Each item in the dictionary should be a two-element list. These lists typically span the entire range of parameter space over which numerical testing takes place. In this way, sets of variables which span the space over which a method is to be tested can be automatically generated.

    Example:
    >>> test_params = {"barrier": [0.5, 5],
    ...                "voltage": [0, 50]}
    >>> combos = gen_combos(test_params)
    >>> print(combos)
    [{'voltage': 0, 'barrier': 0.5}, {'voltage': 0, 'barrier': 5}, {'voltage': 50, 'barrier': 0.5}, {'voltage': 50, 'barrier': 5}]
    """

    # First, I generate a list of all the combinations of values in test_params.
    combs = list(itertools.product(*test_params.itervalues()))

    # Next I iterate over the result and create dicts via zip.
    param_list = []
    for comb in combs:
        partial_params = dict(zip(test_params.keys(), comb))
        param_list.append(partial_params)

    return param_list

def overlay_dicts(overdict, underdict):
    """
    Overlays and concatenates one dict on another; returns resulting dict.

    The keys of the returned dict is the union of the set of keys from both input dicts.

    :param dict overdict: If both overdict and underdict have some keys in common, the resulting dict will have data corresponding to the keys of `overdict`.
    :param dict underdict: Dict to be concatenated with overdict.

    Example:

    >>> a = {"voltage": 0.5, "barrier": 0.2}
    >>> b = {"voltage": 0.75, "richardson": 10}
    >>> cat_dict = overlay_dicts(b, a)
    >>> print cat_dict
    {'barrier': 0.2, 'richardson': 10, 'voltage': 0.5}
    """
    overlay = dict(overdict, **underdict)
    return overlay

def overlay_dict_list_items_on_dict(dict_list, underdict):    
    """
    Overlays each item of a list on a dict and return the resulting list.

    :param list dict_list: List containing dicts which will be overlayed on the dict passed to overlay_dict_on_each_item_of_list. This list must only contain dicts, otherwise this method will fail.
    :param dict underdict: Dict which will be overlaid.
    """
    overlaid_list = []
    for itm in dict_list:
        overlay = overlay_dicts(itm, underdict)
        overlaid_list.append(overlay)

    return overlaid_list





def cat_eqn(partial_params, eqn, key="eqn"):
    """
    Simply adds the eqn item to the dictionary.
    """
    params[key] = eqn
    return params

def eval_rhs(params, key = "eqn"):
    """
    Return the evaluated rhs of the equation.
    """
    split_eqn = params[key].split("==")
    rhs = split_eqn[1].strip()

    return eval(rhs % params)

def add_eqn_output(params, key = "eqn"):
    """
    Evaluate the rhs of the equation, add the value to the dict with the key named by the lhs.

    >>> params = {"a": 3, "b": 4, "eqn": "%(c)s**2 == %(a)s**2 + %(b)s**2}
    >>> add_eqn_output(params)
    {'a': 3, 'b': 4, 'eqn': "%(c)s**2 == %(a)s**2 + %(b)s**2, 'c': 25}
    """
    split_eqn = params[key].split("==")

    # Regular expression to pull the variable name out of the paranthesis.
    regex_lhs_name = re.compile(".*?\((.*?)\)")
    # Get the name of the lhs variable.
    lhs_name = regex_lhs_name.match(split_eqn[0]).group(1)

    params[lhs_name] = eval_rhs(params, key)

    return params

def tf(abscissa, params, abscissa_key, eqn_key = "eqn"):
    """
    Target function for the rootfinding algorithm.
    """
    # This construction is fugly and should be changed.
    target_params = dict(params, **dict([(abscissa_key, abscissa)]))

    rounded_output = round_exponential(eval_rhs(params, eqn_key))
    return eval_rhs(target_params, eqn_key) - rounded_output

def special_case_adjust_param(params, abscissa_key):
    """
    Adjusts a parameter until the special case is achieved.
    """
    root = optimize.newton(tf, params[abscissa_key], args=(params, abscissa_key, "eqn"))

    return dict(params, **{abscissa_key: root})

# code:
# First, set up the parameters over which the test will occur.
test_params = {"barrier": [0.5, 5],
               "voltage": [0, 50]}

eqn = "%(barrier_ht)s == %(electron_charge)s * (%(voltage)s + %(barrier)s)"

param_list = gen_combos(test_params)
adjusted_param_list = []
for params in param_list:
    params = cat_params(params, el)
    params = cat_params(params, physical_constants)
    params = cat_eqn(params, eqn)
    params = add_eqn_output(params)
    params = special_case_adjust_param(params, "voltage")
    adjusted_param_list.append(params)
