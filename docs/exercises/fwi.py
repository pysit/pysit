import numpy as np
import matplotlib.pyplot as plt

from models import basic_model

config = dict()


##############################################################################
# Problem 1.1

def ricker(t, config):

    nu0 = config['nu0']

    # implementation goes here

    return w

# Configure source wavelet
config['nu0'] = 10  # Hz

# Evaluate wavelet and plot it
ts = np.linspace(0, 0.5, 1000)
ws = ricker(ts, config)

plt.figure()
plt.plot(ts, ws,
         color='green',
         label=r'$\nu_0 =\,{0}$Hz'.format(config['nu0']),
         linewidth=2)

plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$w(t)$', fontsize=18)
plt.title('Ricker Wavelet', fontsize=22)

plt.legend()


##############################################################################
# Problem 1.2

def point_source(value, position, config):

    # implementation goes here

    return f

# Domain parameters
config['x_limits'] = [0.0, 1.0]
config['nx'] = 201
config['dx'] = (config['x_limits'][1] - config['x_limits'][0]) / (config['nx']-1)

# Source parameter
config['x_s'] = 0.1


##############################################################################
# Problem 1.3

def construct_matrices(C, config):

    # implementation goes here

    return M, A, K

# Load the model
C, C0 = basic_model(config)

# Build an example set of matrices
M, A, K = construct_matrices(C, config)


##############################################################################
# Problem 1.4

def leap_frog(C, sources, config):

    # implementation goes here

    return us  # list of wavefields

# Set CFL safety constant
config['alpha'] = 1.0/6.0

# Define time step parameters
config['T'] = 3  # seconds
config['dt'] = config['alpha'] * config['dx'] / C.max()
config['nt'] = int(config['T']/config['dt'])

# Generate the sources
sources = list()
for i in xrange(config['nt']):
    t = i*config['dt']
    f = point_source(ricker(t, config), config['x_s'], config)
    sources.append(f)

# Generate wavefields
us = leap_frog(C, sources, config)


##############################################################################
# Problem 1.5

def plot_space_time(us, config, title=None):

    # implementation goes here
    pass

# Call your function
plot_space_time(us, config, title=r'u(x,t)')


##############################################################################
# Problem 1.6

def record_data(u, config):

    # implementation goes here

    return d

# Receiver position
config['x_r'] = 0.15


##############################################################################
# Problem 1.7

def forward_operator(C, config):

    # implementation goes here

    return us, trace


us, d = forward_operator(C, config)

# The last argument False excludes the end point from the list
ts = np.linspace(0, config['T'], config['nt'], False)

plt.figure()
plt.plot(ts, d, label=r'$x_r =\,{0}$'.format(config['x_r']), linewidth=2)
plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$d(t)$', fontsize=18)
plt.title('Trace at $x_r={0}$'.format(config['x_r']), fontsize=22)
plt.legend()


##############################################################################
# Problem 2.1



##############################################################################
# Problem 2.2



##############################################################################
# Problem 2.3

def imaging_condition(qs, u0s, config):

    # implementation goes here

    return image

# Compute the image
I_rtm = imaging_condition(qs, u0s, config)

# Plot the comparison
xs = np.arange(config['nx'])*config['dx']
dC = C-C0

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(xs, dC, label=r'$\delta C$')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(xs, I_rtm, label=r'$I_\text{RTM}$')
plt.legend()


##############################################################################
# Problem 2.4



##############################################################################
# Problem 2.5

def adjoint_operator(C0, d, config):

    # implementation goes here

    return image



##############################################################################
# Problem 3.1

def linear_sources(dm, u0s, config):

    # implementation goes here

    return sources


##############################################################################
# Problem 3.2

def linear_forward_operator(C0, dm, config):

    # implementation goes here

    return u1s


##############################################################################
# Problem 3.3

def adjoint_condition(C0, config):

    # implementation goes here
    pass


##############################################################################
# Problem 4.1

def gradient_descent(C0, d, k, config):

    # implementation goes here

    return sources


##############################################################################
# Problem 3.2

def linear_forward_operator(C0, dm, config):

    # implementation goes here

    return u1s


##############################################################################
# Problem 3.3

def adjoint_condition(C0, config):

    # implementation goes here
    pass


##############################################################################
# Problem 4.1

def gradient_descent(C0, d, k, config):

    # implementation goes here
    pass


##############################################################################
# Problem 4.2

