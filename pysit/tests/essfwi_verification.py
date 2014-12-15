# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import horizontal_reflector

#In this file I set up the simultaneous supershot for Encoded Simultaneous Source FWI (ESSFWI).

if __name__ == '__main__':
    # Setup

    #   Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    m = CartesianMesh(d, 71, 41) 

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)


    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax

    shots = equispaced_acquisition(m,
                                   RickerWavelet(5.0),
                                   sources=10,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   )

    # Define and configure the wave solver
    trange = (0.0,3.0)

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=2,
                                         trange=trange,
                                         kernel_implementation='cpp')

    # Generate synthetic Seismic data
    tt = time.time()
    wavefields =  []
    true_model = solver.ModelParameters(m,{'C': C})
    initial_model = solver.ModelParameters(m,{'C': C0})
    generate_seismic_data(shots, solver, true_model, wavefields=wavefields)

    print 'Data generation: {0}s'.format(time.time()-tt)

    print 'Now that the data is generated, test if ESSFWI objective and gradient have expected values equal to the sequential source versions.'


    essfwi_shot = SourceEncodedSupershot(shots)
    solver.model = initial_model

    objective = TemporalLeastSquares(solver)
    aux_info = {'objective_value': (True, None)}
    sequential_grad = objective.compute_gradient(shots,initial_model,aux_info)
    sequential_objective = aux_info['objective_value'][1]
    
    essfwi_grads = []
    essfwi_objectives = []
    n_iter = 10
    for i in xrange(n_iter):
        aux_info = {'objective_value': (True, None)}
        essfwi_grad = objective.compute_gradient([essfwi_shot],initial_model,aux_info) #list of shots is expected...
        essfwi_objective = aux_info['objective_value'][1]
        essfwi_grads.append(essfwi_grad)
        essfwi_objectives.append(essfwi_objective)
        essfwi_shot.generate_weight_vector() #Generate new weights.
    
    #average the gradients and the objective values
    
    #initialize
    avg_essfwi_grad = 0.0*essfwi_grad
    avg_essfwi_objective = 0.0
    
    for essfwi_grad, essfwi_objective in zip(essfwi_grads, essfwi_objectives):
        avg_essfwi_grad += 1.0/n_iter * essfwi_grad
        avg_essfwi_objective += 1.0/n_iter * essfwi_objective
    
    plt.figure(1) #The sequential shot gradient
    plt.imshow(np.reshape(sequential_grad.data, (41,71), 'F'))
    plt.colorbar()
    plt.figure(2) #The ESSFWI shot gradient
    plt.imshow(np.reshape(essfwi_grad.data, (41,71), 'F'))
    plt.colorbar()
    plt.figure(3) #The averaged ESSFWI shot gradient
    plt.imshow(np.reshape(avg_essfwi_grad.data, (41,71), 'F'))
    plt.colorbar()
    plt.show()
    
    print "DO THE SAME FOR FREQUENCY DOMAIN IMPLEMENTATION."


    
#
#    # Define the inversion algorithm
#    invalg = LBFGS(objective)
#    initial_value = solver.ModelParameters(m,{'C': C0})
#
#    # Execute inversion algorithm
#    print('Running LBFGS...')
#    tt = time.time()
#
#    nsteps = 5
#
#    status_configuration = {'value_frequency'           : 1,
#                            'residual_frequency'        : 1,
#                            'residual_length_frequency' : 1,
#                            'objective_frequency'       : 1,
#                            'step_frequency'            : 1,
#                            'step_length_frequency'     : 1,
#                            'gradient_frequency'        : 1,
#                            'gradient_length_frequency' : 1,
#                            'run_time_frequency'        : 1,
#                            'alpha_frequency'           : 1,
#                            }
#
##   line_search = ('constant', 1e-16)
#    line_search = 'backtrack'
#
#    result = invalg(shots, initial_value, nsteps,
#                    line_search=line_search,
#                    status_configuration=status_configuration, verbose=True)
#
#    print '...run time:  {0}s'.format(time.time()-tt)
#
#    obj_vals = np.array([v for k,v in invalg.objective_history.items()])
#
#    plt.figure()
#    plt.semilogy(obj_vals)
#
#    clim = C.min(),C.max()
#
#    # Do something to visualize the results
#    plt.figure()
#    plt.subplot(3,1,1)
#    vis.plot(C0, m, clim=clim)
#    plt.title('Initial Model')
#    plt.subplot(3,1,2)
#    vis.plot(C, m, clim=clim)
#    plt.title('True Model')
#    plt.subplot(3,1,3)
#    vis.plot(result.C, m, clim=clim)
#    plt.title('Reconstruction')
#
#    plt.show()
