# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import horizontal_reflector

if __name__ == '__main__':

    pml = PML(150, 100)

    npx = 32
    npz = 32
    dx = 30.0
    dz = 30.0

    x_config = (0.0, (npx)*dx, pml, pml)
    z_config = (0.0, (npz)*dz, pml, pml)

    d = RectangularDomain(x_config, z_config)
    m = CartesianMesh(d, npx, npz)

    xmin = d.x.lbound
    xmax = d.x.rbound
    nx   = m.x.n
    zmin = d.z.lbound
    zmax = d.z.rbound

    # Set up shots
    Nshots = 32
    shots = []
    zpos = 0
    xpos = np.linspace(xmin, xmax, nx)
    for i in range(Nshots):

        # Define source location and type
        source = PointSource(m, (xmax*(i+1.0)/(Nshots+1.0), 0.1), RickerWavelet(4.0), approximation='delta')

        # Define set of receivers
        receivers = ReceiverSet(m, [PointReceiver(m, (x, zpos)) for x in xpos])

        # Create and store the shot
        shot = Shot(source, receivers)
        shots.append(shot)

    Clayer1 = np.ones((16,32))*3000.0
    Clayer2 = np.ones((16,32))*4000.0
    Carray = np.append(Clayer1,Clayer2, axis=0)
    C = np.reshape(Carray, (npx*npz,1), 'F')
    C0 = 3000*np.ones((npx*npz,1))

    solver = ConstantDensityHelmholtz(m)
    base_model = solver.ModelParameters(m,{'C': C})
    frequencies = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0] #random bunch of frequencies
    generate_seismic_data(shots, solver, base_model, frequencies=frequencies)

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

    #Do 6 steps at frequencies 2.0 and 3.5 together, and then do 3 steps at frequencies 3.5 and 5.0 together.
    #loop_configuration=[(6,{'frequencies' : [2.0, 3.5]}), (3,{'frequencies' : [3.5, 5.0]})]
    loop_configuration=[(6,{'frequencies' : [1.0]}), (6,{'frequencies' : [1.5, 2.0]}), (6,{'frequencies' : [2.5, 3.0, 3.5]}), (6,{'frequencies' : [4.0, 4.5, 5.0]}), (6,{'frequencies' : [5.0, 7.5, 10.0]})]
    result = invalg(shots, initial_value, loop_configuration, verbose=True, status_configuration=status_configuration)
    print('...run time:  {0}s'.format(time.time()-tt))

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

