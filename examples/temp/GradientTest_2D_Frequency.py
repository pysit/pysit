# Std import block
import time
import copy
import os

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import horizontal_reflector


from GradientTest import GradientTest

if __name__ == '__main__':
    # Setup
    hybrid = False
    # enable Open MP multithread solver
    os.environ["OMP_NUM_THREADS"] = "4"

    #   Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    m = CartesianMesh(d, 91, 71)

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=1,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   )

    # Define and configure the wave solver
    trange = (0.0, 3.0)

    solver_time = ConstantDensityAcousticWave(m,
                                              spatial_accuracy_order=6,
                                              kernel_implementation='omp',
                                              trange=trange)
    # Generate synthetic Seismic data
    print('Generating data...')
    base_model = solver_time.ModelParameters(m, {'C': C})
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
    grad_test = GradientTest(objective)
    grad_test.base_model = solver.ModelParameters(m, {'C': C0})
    grad_test.length_ratio = np.power(5.0, range(-6, -0))

    # Set up the perturbation direction
    dC_vec = copy.deepcopy(grad_test.base_model)
    m_size = m._shapes[(False, True)]
    tmp = np.random.normal(0, 1, m_size)
    tmp = np.ones(m_size)
    tmp[0:2, :] = 0.0
    tmp[m_size[0]-2:m_size[0], :] = 0.0
    tmp[:, 0:2] = 0.0
    tmp[:, m_size[1]-2:m_size[1]] = 0.0
    tmp = np.reshape(tmp, grad_test.base_model.data.shape)
    dC_vec.data = tmp
    norm_dC_vec = np.linalg.norm(dC_vec.data)
    norm_base_model = np.linalg.norm(grad_test.base_model.data)
    dC_vec.data = dC_vec.data * 0.1 * (norm_base_model / norm_dC_vec)
    grad_test.model_perturbation = dC_vec

    # Execute inversion algorithm
    print('Gradient test ...')
    tt = time.time()

    objective_arguments = {'frequencies': [2.0]}

    result = grad_test(shots, objective_arguments)

    print('...run time:  {0}s'.format(time.time()-tt))

    print(grad_test.objective_value)

    plt.figure()
    plt.loglog(grad_test.length_ratio, grad_test.zero_order_difference, 'b',
               grad_test.length_ratio, grad_test.length_ratio, 'r')
    plt.title('Zero order difference')
    plt.gca().legend(('df_0', 'h'))

    plt.figure()
    plt.loglog(grad_test.length_ratio, grad_test.first_order_difference, 'b',
               grad_test.length_ratio, np.power(grad_test.length_ratio, 1.0), 'y',
               grad_test.length_ratio, np.power(grad_test.length_ratio, 2.0), 'r')
    plt.title('First order difference')
    plt.gca().legend(('df_1', 'h', 'h^2'))

    plt.show()

    a = 1
