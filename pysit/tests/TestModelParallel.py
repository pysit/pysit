from mpi4py import MPI

import matplotlib.pyplot as plt
import numpy as np

from pysit import *
from pysit.core.mesh import CartesianMesh
from pysit.gallery import horizontal_reflector
from pysit.modeling.temporal_modeling_parallel import TemporalModelingParallel
from pysit.util.parallel import ParallelWrapCartesianMesh

def test_model_parallel_forward_solver():
    
    # Left and right bounds of global domain
    lbound = 0.1
    rbound = 0.7

    # Left and right boundary conditions of global domain
    bczl = PML(0.1, 100, ftype='quadratic')
    bczr = PML(0.1, 100, ftype='quadratic')

    # Configuration tuple for global domain
    zconfig = (lbound, rbound, bczl, bczr)

    # Create global domain
    domain = RectangularDomain(zconfig)

    # Need to define solver accuracy order and padding before declaring domain
    # decomposed mesh so that ghost boundary conditions can be calculated
    # correctly
    solver_accuracy_order = 2
    solver_padding = solver_accuracy_order // 2

    # Create a parallel mesh
    pwrap = ParallelWrapCartesianMesh(comm=MPI.COMM_WORLD)
    m  = CartesianMesh(domain, 301, solver_padding=solver_padding, pwrap=pwrap)
    
    # Set up horizontal reflector problem
    C, C0, m, d = horizontal_reflector(m)
    
    # Set up source and receiver
    zpos = 0.2
    source = PointSource(m, (zpos), RickerWavelet(25.0))
    receiver = PointReceiver(m, (zpos))

    shot = Shot(source, receiver)
    shots = list()
    shots.append(shot)

    # Define and configure the wave solver
    trange = (0.0,3.0)
    solver = ConstantDensityAcousticWave(m,
                                         formulation='scalar',
                                         kernel_implementation='numpy',
                                         spatial_accuracy_order=solver_accuracy_order,
                                         trange=trange)

    print('Generating data...')
    wavefields = []
    base_model = solver.ModelParameters(m, {'C': C})
    generate_seismic_data(shots, solver, base_model, wavefields=wavefields)

    tools = TemporalModelingParallel(solver)
    m0 = solver.ModelParameters(m, {'C': C0})

    fwdret = tools.forward_model(shot, m0, return_parameters=['wavefield', 'dWaveOp', 'simdata'])
    dWaveOp0 = fwdret['dWaveOp']
    inc_field = fwdret['wavefield']
    data = fwdret['simdata']

    # Getthe incedent field as an array
    field = np.array(inc_field)
    field = np.squeeze(field)
    field = np.flip(field, axis=0)


    data = {'field':field} 
    data = pwrap.comm.gather(data, root=0)

    if pwrap.rank == 0:
        
        field_all = field
        for i in range(1, pwrap.size):

            field_curr = data[i]['field']
            if field_curr.shape[0] != field_all.shape[0]:
                field_curr = field_curr[:-1,:]

            field_all = np.hstack((field_all, field_curr))
        
        plt.figure()
        plt.imshow(field_all, cmap='gray')
        ax = plt.gca()
        ax.set_aspect(0.05)
        
        plt.show()

if __name__ == '__main__':
    test_model_parallel_forward_solver()
