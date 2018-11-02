# Std import block
import time
import copy

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import marmousi
from pysit.gallery import marmousi2

if __name__ == '__main__':
    #   Load or generate true wave speed
    C, C0, m, d = marmousi(patch='mini_square')
#   C, C0, m, d = marmousi2(patch='mini_square')

    # Set up shots
    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=1,
                                   source_depth=500.0,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_kwargs={},
                                   )

    # Define and configure the wave solver
    trange = (0.0, 3.0)

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=6,
                                         trange=trange,
                                         kernel_implementation='cpp')

    # Generate synthetic Seismic data
    print('Generating data...')
    base_model = solver.ModelParameters(m,{'C': C})
    tt = time.time()
    wavefields = []
    generate_seismic_data(shots, solver, base_model, wavefields=wavefields)
    print('Data generation: {0}s'.format(time.time()-tt))

    objective = TemporalLeastSquares(solver)

    # Define the inversion algorithm
    invalg = LBFGS(objective)
    initial_value = solver.ModelParameters(m,{'C': C0})

    # Execute inversion algorithm
    print('Running Descent...')
    tt = time.time()

    nsteps = 5
    status_configuration = {'value_frequency'           : 1,
                     'residual_length_frequency' : 1,
                     'objective_frequency'       : 1,
                     'step_frequency'            : 1,
                     'step_length_frequency'     : 1,
                     'gradient_frequency'        : 1,
                     'gradient_length_frequency' : 1,
                     'run_time_frequency'        : 1,
                     'alpha_frequency'           : 1,
                     }

    line_search = 'backtrack'

    result = invalg(shots, initial_value, nsteps,
                    line_search=line_search,
                    status_configuration=status_configuration, verbose=True)

    print('...run time:  {0}s'.format(time.time()-tt))

    obj_vals = np.array([v for k,v in list(invalg.objective_history.items())])

    plt.figure()
    plt.semilogy(obj_vals)

    clim = C.min(),C.max()

    # Do something to visualize the results
    plt.figure()
    plt.subplot(3,1,1)
    vis.plot(C0, m, clim=clim)
    plt.title('Initial Model')
    plt.subplot(3,1,2)
    vis.plot(C, m, clim=clim)
    plt.title('True Model')
    plt.subplot(3,1,3)
    vis.plot(result.C, m, clim=clim)
    plt.title('Reconstruction')

    plt.figure()
    plt.subplot(2,1,1)
    vis.plot(result.C-C0, m)
    plt.title('Recon - Initial')
    plt.subplot(2,1,2)
    vis.plot(C-result.C, m)
    plt.title('True-Recon')

    plt.show()


