# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import horizontal_reflector
from pysit.gallery import marmousi
from pysit.gallery import marmousi2

if __name__ == '__main__':
    # Setup

    #   Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    # The first 2 entires in these tuples indicates the physical domain dimensions we will use.
    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)
    # nx and nz specify the number of nodes used in the computation.
    nx = 91
    nz = 71
    m = CartesianMesh(d, nx, nz)
    C, C0, m, d = horizontal_reflector(m)  # C has two reflectors at depth. 
    
    # Generate the Model Parameters in terms of Kappa and Rho (with 2 reflectors at depth).
    # To do this we "split up" the two reflectors contained in dC.
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

    # Model parameters have to be column vectors. We adjusted their shape above for easier manipulation.
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
    zpos = zmin + (1./9.)*zmax

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=5,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   )

    # Define and configure the wave solver
    trange=(0.0,3.0)

    # For the frequency inversion, we will do 5 optimization steps for the frequency set [1,2,3,4], 5 for the set [5, 6, 7, 8] etc.
    # It is important to break up the inversion procedure into "bundles" of similar magnitude frequencies because the gradient of our objective
    # has an omega**2 term. (so large frequencies will dwarf smaller frequencies if their magnitudes are too different).
    # Note: Pysit will be able to solve the Helmholtz equation significantly faster if you downlaoad and implement petsc.
    loop_configuration=[(5,{'frequencies' : [1.0,2.0,3.0,4.0]}), (5,{'frequencies' : [5.0, 6.0, 7.0, 8.0]}), (5,{'frequencies' : [9.0, 10.0, 11.0, 12.0]}), (5,{'frequencies' : [13.0, 14.0, 15.0, 16.0]})]
    
    solver_time = VariableDensityAcousticWave(m, 
                                         formulation='scalar',
                                         model_parameters=model_param, 
                                         spatial_accuracy_order=2,
                                         trange=trange)
    solver = VariableDensityHelmholtz(m,
                                      spatial_accuracy_order=4)
    # Generate synthetic Seismic data
    print('Generating data...')
    base_model = solver_time.ModelParameters(m,model_param)
    tt = time.time()
    
    # The data we are using to calculate our residual is computed in the time domain, and we take a DFT of it
    # such that we can compare our frequency reciever data against it.
    generate_seismic_data(shots, solver_time, base_model) 
    print('Data generation: {0}s'.format(time.time()-tt))

    # Define and configure the objective function
    objective = FrequencyLeastSquares(solver)

    # Define the inversion algorithm
    invalg = LBFGS(objective)
    initial_value = solver.ModelParameters(m,model_init)

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

    # result = invalg(shots, initial_value, loop_configuration, verbose=True, status_configuration=status_configuration, petsc='mkl_pardiso')
    result = invalg(shots, initial_value, loop_configuration, verbose=True, status_configuration=status_configuration)
    
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
