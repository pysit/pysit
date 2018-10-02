import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from pysit import *
from pysit.gallery import marmousi

if __name__ == '__main__':
    C, C0, m, d = marmousi(patch='mini_square')
    delta_vel = 1.0            #Positive numbers indicate that estimate C0 is slower than the true velocity.
    C = C0 + delta_vel
    
    dx = m.x.delta
    dz = m.z.delta
    
    x_min = d.x.lbound
    x_max = d.x.rbound
    
    z_min = d.z.lbound
    z_max = d.z.rbound

    shots = []
    
    source_pos   = (x_min + 25*dx, z_min + 5* dz)
    receiver_pos = (x_max - 25*dx, z_min + 5* dz)
    
    peakfreq = 15.0
    source    = PointSource(m, source_pos, RickerWavelet(peakfreq), approximation = 'delta')
    receivers = ReceiverSet(m, [PointReceiver(m, receiver_pos, approximation = 'delta')], approximation = 'delta') 
    shot      = Shot(source, receivers)
    shots.append(shot)
    
    # Define and configure the wave solver
    trange = (0.0, 7.0) #Long time because source and receiver far away from each other 

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=6,
                                         trange=trange,
                                         kernel_implementation='cpp')
    
    # Generate synthetic Seismic data
    tt = time.time()
    initial_model = solver.ModelParameters(m,{'C': C0})
    true_model    = solver.ModelParameters(m,{'C':  C})
    generate_seismic_data(shots, solver, true_model)

    print('Data generation: {0}s'.format(time.time()-tt))
        
    tt = time.time()
    
    dt = solver.dt
    dominant_period = 1.0/peakfreq
    imaging_steps_per_dominant_period = 15
    timesteps_per_dominant_period = dominant_period/dt
    imaging_period = np.floor(timesteps_per_dominant_period/imaging_steps_per_dominant_period)
    
    aux_info_1 = {'pseudo_hess_diag': (True, None)}; aux_info_2 = {'pseudo_hess_diag': (True, None)}; aux_info_3 = {'pseudo_hess_diag': (True, None)}

    objective_1 = TemporalLeastSquares(solver, imaging_period = 1)
    objective_2 = TemporalLeastSquares(solver, imaging_period = imaging_period)
    objective_3 = TemporalLeastSquares(solver, imaging_period = 2.0*imaging_period)
        
    grad1      = objective_1.compute_gradient(shots,initial_model, aux_info = aux_info_1)
    print('generated gradient using normal imaging period')
    grad2      = objective_2.compute_gradient(shots,initial_model, aux_info = aux_info_2)
    print('generated gradient using reduced imaging period \n')
    grad3      = objective_3.compute_gradient(shots,initial_model, aux_info = aux_info_3)
    print('generated gradient using further reduced imaging period \n')
    print('Gradient generation: {0}s'.format(time.time()-tt))
    
    pseudo_hess_diag_1 = aux_info_1['pseudo_hess_diag'][1]; pseudo_hess_diag_2 = aux_info_2['pseudo_hess_diag'][1]; pseudo_hess_diag_3 = aux_info_3['pseudo_hess_diag'][1]
    
    rel_dif_grad_2 = np.linalg.norm(grad2.data - grad1.data)/np.linalg.norm(grad1.data)
    rel_dif_grad_3 = np.linalg.norm(grad3.data - grad1.data)/np.linalg.norm(grad1.data)
    
    rel_diff_pseudo_2 = np.linalg.norm(pseudo_hess_diag_2 - pseudo_hess_diag_1)/np.linalg.norm(pseudo_hess_diag_1)
    rel_diff_pseudo_3 = np.linalg.norm(pseudo_hess_diag_3 - pseudo_hess_diag_1)/np.linalg.norm(pseudo_hess_diag_1)
    
    print(rel_dif_grad_2, rel_dif_grad_3, rel_diff_pseudo_2, rel_diff_pseudo_3)  