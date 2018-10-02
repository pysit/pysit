# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt
import os

from pysit import *
from pysit.gallery import horizontal_reflector

if __name__ == '__main__':
    # Setup
    hybrid=False
    os.environ["OMP_NUM_THREADS"] = "4"
    
    #   Define Domain
    # PML with auxilary fields
    # pmlx = PML(0.1, 100)
    # pmlz = PML(0.1, 100)
    # PML without auxilary fields
    pmlx = PML(0.1, 100,compact=True)
    pmlz = PML(0.1, 100,compact=True)

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    m = CartesianMesh(d, 100, 100)

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=4,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   )

    # Define and configure the wave solver
    trange = (0.0,3.0)

    # solver_time = ConstantDensityAcousticWave(m,
    #                                           spatial_accuracy_order=6,
    #                                           kernel_implementation='omp',
    #                                           trange=trange)
    
    solver = ConstantDensityHelmholtz(m)
    frequencies = [2.0, 3.5, 5.0]

    # Generate synthetic Seismic data
    print('Generating data...')
    base_model = solver.ModelParameters(m,{'C': C})
    tt = time.time()
    generate_seismic_data(shots, solver, base_model, frequencies=frequencies, petsc='mumps')
    # generate_seismic_data_from_file(shots,save_method='h5py')
    print('Data generation: {0}s'.format(time.time()-tt))

    # Define and configure the objective function
    if hybrid:
        solver = ConstantDensityAcousticWave(m,
                                             spatial_accuracy_order=4,
                                             trange=trange)
        objective = HybridLeastSquares(solver)
    else:

        solver = ConstantDensityHelmholtz(m,
                                          spatial_accuracy_order=4)
        objective = FrequencyLeastSquares(solver)

    # Define the inversion algorithm
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
    invalg.max_linesearch_iterations=18


    loop_configuration=[(4,{'frequencies' : [2.0, 3.5, 5.0]})]

    
    result = invalg(shots, initial_value, loop_configuration, verbose=True, status_configuration=status_configuration, petsc='mumps')
    # result = invalg(shots, initial_value, loop_configuration, verbose=True, status_configuration=status_configuration)
    print('...run time:  {0}s'.format(time.time()-tt))

    obj_vals = np.array([v for k,v in list(invalg.objective_history.items())])

    plt.figure()
    plt.semilogy(obj_vals)

    plt.figure()
    plt.subplot(3,1,1)
    vis.plot(C0, m)
    plt.title('Initial Model')
    plt.subplot(3,1,2)
    vis.plot(C, m)
    plt.title('True Model')
    plt.subplot(3,1,3)
    vis.plot(result.C, m)
    plt.title('Reconstruction')

    plt.show()

