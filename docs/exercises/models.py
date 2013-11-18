from __future__ import division

import numpy as np


def basic_model(config):
    """Returns 1D derivative of Gaussian reflector model.

    Given a configuration dictionary that specifies:

    1) 'x_limits': A tuple specifying the left and right bound of the domain
    2) 'nx': The number of nodes or degrees of freedom, including the end
             points
    3) 'dx': The spatial step

    returns a 2-tuple containing the true velocity model C and the background
    velocity C0.

    """

    x_limits = config['x_limits']
    nx = config['nx']
    dx = config['dx']

    xs = x_limits[0] + np.arange(nx)*dx

    C0 = np.ones(nx)

    dC = -100.0*(xs-0.5)*np.exp(-((xs-0.5)**2)/(1e-4))
    dC[np.where(abs(dC) < 1e-7)] = 0

    C = C0 + dC

    return C, C0


def gauss_model(config):

    x_limits = config['x_limits']
    nx = config['nx']
    dx = config['dx']

    xs = x_limits[0] + np.arange(nx)*dx

    C0 = np.ones(nx)

    dC = 0.2*np.exp(-((xs-0.4)**2)/(1e-4)) - 0.3*np.exp(-((xs-0.55)**2)/(1e-3))
    dC[np.where(abs(dC) < 1e-7)] = 0

    C = C0 + dC

    return C, C0
