import numpy as np

################################################################################
################### Linear and Non-Linear Functions ############################
################################################################################

# These functions implement linear or non-linear helpers.
#
# They are used to:
# 1. Calculate parameters given equalities
# 2. Produce values given input parameters
# 3. Implement caps on output values
#
# Nothing fancy...


def get_linear_params(p1, p2):
    """
    Solves:
        y = ax + b

    Given:
        Points p1 & p2

    Where:
        p1[1] = a*p1[0]+b
        p2[1] = a*p2[0]+b

    And:
        x: # of compute units
        y: CUTTM (Compute Unit To Token Multiplier)

    """

    a = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p2[1] - a * p2[0]

    return a, b


def calc_linear_param(daily_relays, a, b, bootstrap_end, bootstrap_start=0, max_param=[], min_param=[]):
    """
    Applies a piece-wise linear function given the parameters and the limits.
    """
    daily_relays = np.clip(daily_relays, a_max=bootstrap_end, a_min=bootstrap_start)
    param = daily_relays * a + b
    if len(max_param) > 0:
        param = np.clip(param, a_max=max_param, a_min=None)
    if len(min_param) > 0:
        param = np.clip(param, a_max=None, a_min=min_param)
    return param


def get_non_linear_params(p1, p2):
    """
    Solves:
         y = a*x + b/x

     Given:
         Points p1 & p2

     Where:
         p1[1] = a*p1[0]+b/p1[0] = a + b/p1[0]
         p2[1] = a*p2[0]+b/p2[0] = a + b/p2[0]

    And:
        x: # of compute units
        y: CUTTM (Compute Unit To Token Multiplier)

    Solution:
        p1[1] = a + b/p1[0]
        a = p1[1] - b/p1[0]
        p2[1] = p1[1] - b/p1[0] + b/p2[0]
        b = p1[0]*p2[0]*(p2[1]-p1[1]) / (p1[0]-p2[0])
    """

    b = p1[0] * p2[0] * (p2[1] - p1[1]) / (p1[0] - p2[0])
    a = p1[1] - b / p1[0]

    return a, b


def calc_non_linear_param(daily_relays, a, b, bootstrap_end, bootstrap_start=0, max_param=[], min_param=[]):
    """
    Applies a piece-wise non-linear function given the parameters and the limits.
    """

    daily_relays = np.clip(daily_relays, a_max=bootstrap_end, a_min=bootstrap_start)

    param = (daily_relays * a + b) / daily_relays

    if len(max_param) > 0:
        param = np.clip(param, a_max=max_param, a_min=None)
    if len(min_param) > 0:
        param = np.clip(param, a_max=None, a_min=min_param)
    return param
