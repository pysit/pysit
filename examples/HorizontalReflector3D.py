# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import horizontal_reflector

if __name__ == '__main__':
    # Setup

    #   Define Domain
    pmlx = PML(0.1, 100)
    pmly = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 1.0, pmlx, pmlx)
    y_config = (0.1, 0.9, pmly, pmly)
    z_config = (0.1, 0.8, pmlz, pmlz)

#   bc = Dirichlet()
#   x_config = (0.1, 1.0, bc, bc)
#   z_config = (0.1, 0.8, bc, bc)

    d = RectangularDomain(x_config, y_config, z_config)

    m = CartesianMesh(d, 46, 41, 36)

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax

    nsources = 2,3
    nreceivers = 30, 30
    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=nsources,
                                   source_depth=0.1,
                                   source_kwargs={},
                                   receivers=nreceivers,
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   )

    # Define and configure the wave solver
    trange = (0.0,3.0)

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
    invalg = GradientDescent(objective)
    initial_value = solver.ModelParameters(m,{'C': C0})

    # Execute inversion algorithm
    print('Running LBFGS...')
    tt = time.time()

    nsteps = 3

    configuration = {'value_frequency'           : 1,
                     'residual_frequency'        : 1,
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
                    status_configuration=configuration, verbose=True)

    print('Run time:  {0}s'.format(time.time()-tt))

    plt.figure()
    vis.plot(C, m)

    plt.figure()
    vis.plot(result.C, m)

    plt.show()
