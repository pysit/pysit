# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery.horizontal_reflector import horizontal_reflector

if __name__ == '__main__':
    # Setup

    #   Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    # The first 2 entires in these tuples indicate the physical domain size.
    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)
    
    # nx and nz specify the number of nodes used in the computational mesh.
    nx = 91
    nz = 71
    m = CartesianMesh(d, nx, nz)
    C, C0, m, d = horizontal_reflector(m)  # C has two reflectors at depth. 
    
    # This generate the Model Parameters in terms of Kappa and Rho (with 2 reflectors at depth).
    # We "split up" the two reflectors contained in dC so each can be manipulated seperatly. 
    C0 = np.ones((nx,nz))
    dC = C.reshape(nx,nz) - C0
    dC1 = dC[:,:np.ceil(nz/2)]
    dC2 = dC[:,np.ceil(nz/2):]
    dK = np.zeros((nx,nz))
    dR = np.zeros((nx,nz))
    
    # These next 4 lines allow us to modify the sign and magntiude of each of the bumps for kappa and rho.
    dK[:,:np.ceil(nz/2)] += dC1   
    dR[:,:np.ceil(nz/2)] += dC1   
    dK[:,np.ceil(nz/2):] += dC2
    dR[:,np.ceil(nz/2):] += dC2

    # Model parameters have to be column vectors. We adjusted their shape above simply for easier manipulation.
    kappa = (C0 + dK).reshape((nx*nz,1))
    rho = (C0 + dR).reshape((nx*nz,1))
    C0 = C0.reshape((nx*nz,1))

    kappa0 = C0
    rho0 = C0
    
    # Here we delcare what our true model parameters will be and their initial guesses. 
    model_param = {'kappa': kappa, 'rho': rho}
    model_init = {'kappa' : kappa0, 'rho': rho0}

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax   # the position of the sources will be at 1/9th the total depth (ie. length of z-axis) of the domain.

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=3,     # increasing the number of sources increases the amount of data we are optimizing over. Thus, this improves 
                                   source_depth=zpos,  # accuracy, but also slows down computation time.
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   )

    # Define and configure the wave solver
    trange = (0.0,3.0)   # this tuple defines the time window over which will run the computation.

    solver = VariableDensityAcousticWave(m,                                 # Because their are 2 parameters- kappa and rho,
                                         spatial_accuracy_order=4,          # We use the VariableDensityAcousticWave solver.
                                         trange=trange)

    # Generate synthetic Seismic data
    tt = time.time()
    wavefields =  []
    base_model = solver.ModelParameters(m,model_param)
    generate_seismic_data(shots, solver, base_model, wavefields=wavefields)   # This step uses the TRUE MODEL to compute the data the true model would produce at the recievers. 
                                                                              # We then optimize our "trial" models such that our optimized models produce the same data at the recievers as nearly as possible (using the adjoint state method).
    print('Data generation: {0}s'.format(time.time()-tt))

    objective = TemporalLeastSquares(solver)

    # Define the inversion algorithm
    invalg = LBFGS(objective)
    initial_value = solver.ModelParameters(m,model_init)

    # Execute inversion algorithm
    print('Running LBFGS...')
    tt = time.time()

    # This specifies how many optimization steps we will take (the more we use, the more convergence should be obtained)
    nsteps = 10

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

#   line_search = ('constant', 1e-16)
    line_search = 'backtrack'

    # when we *call* the optimization, LBFGS class which has been initilized as invalg, with the below input, the optimization routine is carried out from our initial guess.
    # The new inverted models will be members in result. ie, to plot the inverted model for kappa or rho, we plot, result.kappa and result.rho.
    result = invalg(shots, initial_value, nsteps,
                    line_search=line_search,
                    status_configuration=status_configuration, verbose=True)

    print('...run time:  {0}s'.format(time.time()-tt))
    
    obj_vals = np.array([v for k,v in list(invalg.objective_history.items())])
    plt.figure()
    plt.semilogy(obj_vals)

    clim = kappa.min(),kappa.max()
    plt.figure()
    plt.subplot(3,1,1)
    vis.plot(C0, m, clim=clim)
    plt.title('Initial Model Kappa')
    plt.subplot(3,1,2)
    vis.plot(kappa, m, clim=clim)
    plt.title('True Model Kappa')
    plt.subplot(3,1,3)
    vis.plot(result.kappa, m, clim=clim)
    plt.title('Reconstruction Kappa')

    plt.figure()
    plt.subplot(3,1,1)
    vis.plot(C0, m, clim=clim)
    plt.title('Initial Model Rho')
    plt.subplot(3,1,2)
    vis.plot(rho, m, clim=clim)
    plt.title('True Model Rho')
    plt.subplot(3,1,3)
    vis.plot(result.rho, m, clim=clim)
    plt.title('Reconstruction Rho')

    plt.show()

