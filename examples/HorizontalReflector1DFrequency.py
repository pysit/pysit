# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import horizontal_reflector

if __name__ == '__main__':
    # Setup
    hybrid=False

    #   Define Domain
    pmlz = PML(0.1, 100, ftype='quadratic')

#   pmlz = Dirichlet()

    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(z_config)

    m = CartesianMesh(d, 301)

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    Nshots = 1
    shots = []

    # Define source location and type
    zpos = 0.2
    source = PointSource(m, (zpos), RickerWavelet(25.0))

    # Define set of receivers
    receiver = PointReceiver(m, (zpos))
    # receivers = ReceiverSet([receiver])

    # Create and store the shot
    shot = Shot(source, receiver)
    # shot = Shot(source, receivers)
    shots.append(shot)

    # Define and configure the wave solver
    trange = (0.0,3.0)

    solver_time = ConstantDensityAcousticWave(m,
                                              spatial_accuracy_order=6,
                                              trange=trange)
    # Generate synthetic Seismic data
    print('Generating data...')
    base_model = solver_time.ModelParameters(m,{'C': C})
    tt = time.time()
    generate_seismic_data(shots, solver_time, base_model)
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
    invalg.max_linesearch_iterations=40

    loop_configuration=[(15,{'frequencies' : [2.0, 3.5, 5.0]}), (15,{'frequencies' : [6.5, 8.0, 9.5]})] #3 steps at one set of frequencies and 3 at another set

    loop_configuration=[(6,{'frequencies' : [2.0, 3.5, 5.0]})]

    result = invalg(shots, initial_value, loop_configuration, verbose=True, status_configuration=status_configuration)

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
