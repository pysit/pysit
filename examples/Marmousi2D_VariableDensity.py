# Std import block
import time 
import os
import pickle # for shot saving
import shutil # for shots saved folder deletion
import numpy as np
import matplotlib.pyplot as plt
from pysit.gallery import marmousi
from pysit import *
from pysit.gallery.horizontal_reflector import horizontal_reflector

""" This variable density marmousi inversion example requires petsc to be installed on your computer in  
 addition to the standard Pysit package """ 

if __name__ == '__main__':
    #setting the num of threads for the use of petsc
    os.environ["OMP_NUM_THREADS"] = "12"

    #Compute one time the data generation for a certain setup
    
    # Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    C, C0, m, d = marmousi(patch='mini_square')

    # To save computation time, we sample the imported ~(450x150) node marmousi model by taking every third x node,
    # and every other z node. 
    C = C.reshape(m.x.n,m.z.n)
    Nx = len(list(range(0,m.x.n,3)))
    C0 = C0.reshape(m.x.n,m.z.n)
    Nz = len(list(range(0,m.z.n,2)))
    print("new shape", Nx, Nz)
    C_ = np.zeros((Nx,Nz))
    C0_ = np.zeros((Nx,Nz))
    for i,j in zip(list(range(0, m.x.n, 3)),list(range(Nx))):
        v = C[i,:]
        v_ = v[list(range(0,m.z.n,2))]
        v0 = C0[i,:]
        v0_ = v0[list(range(0,m.z.n,2))]
        C_[j,:] = v_
        C0_[j,:] = v0_
    C = C_.reshape((Nx*Nz,1))
    C0 = C0_.reshape((Nx*Nz,1))

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 1.0, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)
    m = CartesianMesh(d, Nx, Nz)

    x = np.mean(C)**-1    
    C *=x
    C0 *=x
    dC = C-C0

    rho0 = C0
    kappa0 = C0
    rho = C0 + 0.8*dC
    kappa = C0 + 1.2*dC


    model_param={'kappa': kappa, 'rho': rho}
    model_init={'kappa': kappa0, 'rho': rho0}

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=25,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   )

    # Define and configure the wave solver
    trange = (0.0,3.0)

    solver_time = VariableDensityAcousticWave(m, 
                                         formulation='scalar',
                                         model_parameters=model_param, 
                                         spatial_accuracy_order=2,
                                         trange=trange)

    solver = VariableDensityHelmholtz(m,
                                    model_parameters=model_param,
                                    spatial_shifted_differences=False,
                                    spatial_accuracy_order=2)

    base_model = solver_time.ModelParameters(m,model_param)
    generate_seismic_data(shots, solver_time, base_model) 
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
    invalg.max_linesearch_iterations=25

    loop_configuration=[(30,{'frequencies' : [1,1.5,2.0,2.5] }),(30,{'frequencies' : [3.0,3.5,4.0,4.5] }), (20,{'frequencies' : [5,5.5,6.0,6.5] }),(20,{'frequencies' : [7.0,7.5,8.0,8.5] }), (20,{'frequencies' : [9,10,11,12] }), (20,{'frequencies' : [13,13.5,14,14.5]}),(20,{'frequencies' : [15,16,17,18]}), (20,{'frequencies' : [19,20,21,22,23,24,25,26,27,28,29,30]})] #3 steps at one set of frequencies and 3 at another set

    #result = invalg(shots, initial_value, loop_configuration, verbose=True, status_configuration=status_configuration, petsc='mkl_pardiso')
    result = invalg(shots, initial_value, loop_configuration, verbose=True, status_configuration=status_configuration, petsc='mumps')
    print('...run time:  {0}s'.format(time.time()-tt))

    obj_vals = np.array([v for k,v in list(invalg.objective_history.items())])

    plt.figure()
    plt.semilogy(obj_vals)

    clim = kappa.min(), kappa.max()
    plt.figure()    
    plt.subplot(3,1,1)
    vis.plot(kappa, m, clim=clim)
    plt.title('Kappa True . Normalized')
    plt.colorbar()
    plt.subplot(3,1,2)
    vis.plot(result.kappa, m, clim=clim)
    plt.title('Kappa Recon')
    plt.colorbar()
    plt.subplot(3,1,3)
    vis.plot(kappa0, m, clim=clim)
    plt.title('Kappa Init')
    plt.colorbar()

    clim = rho.min(), rho.max()
    plt.figure()    
    plt.subplot(3,1,1)
    vis.plot(rho, m, clim=clim)
    plt.title('Rho True')
    plt.colorbar()
    plt.subplot(3,1,2)
    vis.plot(result.rho, m, clim=clim)
    plt.title('Rho Recon')
    plt.colorbar()
    plt.subplot(3,1,3)
    vis.plot(rho0, m, clim=clim)
    plt.title('Rho Init')
    plt.colorbar()

    plt.figure()    
    plt.subplot(3,1,1)
    vis.plot(result.kappa-kappa0, m)
    plt.title('Change from Initial, Kappa')
    plt.colorbar()
    plt.subplot(3,1,2)
    vis.plot(kappa-result.kappa, m)
    plt.title('Error from True, Kappa')
    plt.colorbar()

    plt.figure()    
    plt.subplot(3,1,1)
    vis.plot(result.rho-rho0, m)
    plt.title('Change from Initial, Rho')
    plt.colorbar()
    plt.subplot(3,1,2)
    vis.plot(rho-result.rho, m)
    plt.title('Error from True, Rho')
    plt.colorbar()

    plt.show()


