# Std import block
import time
import copy
import os

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import marmousi
from pysit.gallery import marmousi2

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "14"
    #   Load or generate true wave speed we can add the compact operator
    #   flag it will speed up the resolution
    
    # uses a compact PML formulation
    C, C0, m, d = marmousi(patch='mini_square', compact = True)
    # C, C0, m, d = marmousi2(patch='mini_square', compact = True)
    # C, C0, m, d = marmousi(patch='mini_square')
    # C, C0, m, d = marmousi2(patch='mini_square')

    # Set up shots
    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=20,
                                   source_depth=500.0,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_kwargs={},
                                   )

    # Define and configure the wave solver
    solver = ConstantDensityHelmholtz(m)
    frequencies = [2.0, 3.5, 5.0]

    # Generate synthetic Seismic data
    print('Generating data...')
    base_model = solver.ModelParameters(m,{'C': C})
    tt = time.time()
    # generate_seismic_data(shots, solver, base_model, frequencies=frequencies, petsc='mkl_pardiso')
    generate_seismic_data(shots, solver, base_model, frequencies=frequencies, petsc='mumps')
    # generate_seismic_data_from_file(shots,save_method='h5py')
    print('Data generation: {0}s'.format(time.time()-tt))

    objective = FrequencyLeastSquares(solver)

    invalg = LBFGS(objective)
    initial_value = solver.ModelParameters(m,{'C': C0})

    # Execute inversion algorithm
    print('Running Descent...')
    tt = time.time()

    status_configuration = {'value_frequency'           : 1,
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
    invalg.max_linesearch_iterations=40


    loop_configuration=[(20,{'frequencies' : [2.0, 3.5, 5.0]})]

    # result = invalg(shots, initial_value, loop_configuration, verbose=True, status_configuration=status_configuration, petsc='mkl_pardiso')
    result = invalg(shots, initial_value, loop_configuration, verbose=True, status_configuration=status_configuration, petsc='mumps')
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


