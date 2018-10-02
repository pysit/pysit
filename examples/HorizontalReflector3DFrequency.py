# Std import block
import time
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import os

from pysit import *
from pysit.gallery import horizontal_reflector

if __name__ == '__main__':
    # Setup
    os.environ["OMP_NUM_THREADS"] = "16"
    
    #   Define Domain with compact PML
    pmlx = PML(0.1, 100,compact=True)
    pmly = PML(0.1, 100,compact=True)
    pmlz = PML(0.1, 100,compact=True)

    x_config = (0.1, 1.0, pmlx, pmlx)
    y_config = (0.1, 0.9, pmly, pmly)
    z_config = (0.1, 0.8, pmlz, pmlz)


    d = RectangularDomain(x_config, y_config, z_config)

    m = CartesianMesh(d, 45, 40, 35)

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
    solver = ConstantDensityHelmholtz(m)
    frequencies = [2.0,3.5] 

    # Generate synthetic Seismqc data
    print('Generating data...')
    base_model = solver.ModelParameters(m,{'C': C})
    tt = time.time()
    wavefields = []
    generate_seismic_data(shots, solver, base_model, frequencies=frequencies, petsc='mumps')
    # generate_seismic_data(shots, solver, base_model, frequencies=frequencies, petsc='mkl_pardiso')
    
    print('Data generation: {0}s'.format(time.time()-tt))


    solver = ConstantDensityHelmholtz(m,
                                          spatial_accuracy_order=2)
    
    objective = FrequencyLeastSquares(solver)

    # Define the inversion algorithm
    invalg = LBFGS(objective)
    initial_value = solver.ModelParameters(m,{'C': C0})

    # Execute inversion algorithm
    print('Running LBFGS...')
    tt = time.time()


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

    invalg.max_linesearch_iterations=12
    loop_configuration=[(10,{'frequencies' : [2.0,3.5]})]

    result = invalg(shots, initial_value, loop_configuration,
                    status_configuration=configuration, verbose=True, petsc='mumps')
    # result = invalg(shots, initial_value, loop_configuration,
    #                 status_configuration=configuration, verbose=True, petsc='mkl_pardiso')

    print('Run time:  {0}s'.format(time.time()-tt))

    plt.figure()
    vis.plot(C, m)
    plt.savefig("C.png")

    plt.figure()
    vis.plot(result.C, m)
    plt.savefig("result.png")

