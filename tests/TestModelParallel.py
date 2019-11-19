from mpi4py import MPI

import matplotlib.pyplot as plt
import numpy as np

from pysit import *
from pysit.core.mesh import ParallelCartesianMesh
from pysit.gallery import horizontal_reflector

def test_model_parallel_forward_solver():
    
    # Left and right bounds of global domain
    lbound_global = 0.1
    rbound_global = 0.7

    # Left and right boundary conditions of global domain
    bczl_global = PML(0.1, 100, ftype='quadratic')
    bczr_global = PML(0.1, 100, ftype='quadratic')

    # Configuration tuple for global domain
    zconfig_global = (lbound_global, rbound_global, bczl_global, bczr_global)

    # Create global domain
    domain_global = RectangularDomain(zconfig_global)

    # Need to define solver accuracy order and padding before declaring domain
    # decomposed mesh so that ghost boundary conditions can be calculated
    # correctly
    solver_accuracy_order = 2
    solver_padding = solver_accuracy_order // 2

    # Domain decomposition for mesh
    n_nodes_global = 301
    comm = MPI.COMM_WORLD
    pm = ParallelCartesianMesh(domain_global, solver_padding, comm,
            n_nodes_global)
    
    if pm.rank == 0:

        m = pm.mesh_local

        C, C0, m, d = horizontal_reflector(m)

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
                                             spatial_accuracy_order=solver_accuracy_order,
                                             trange=trange)

        np.random.seed(1)
        print('Generating data...')
        wavefields = []
        base_model = solver.ModelParameters(m, {'C': C})
        generate_seismic_data(shots, solver, base_model, wavefields=wavefields)

        tools = TemporalModeling(solver)
        m0 = solver.ModelParameters(m, {'C': C0})

        m1 = m0.perturbation()
        m1 += np.random.rand(*m1.data.shape)

        fwdret = tools.forward_model(shot, m0, return_parameters=['wavefield', 'dWaveOp', 'simdata'])
        dWaveOp0 = fwdret['dWaveOp']
        inc_field = fwdret['wavefield']
        data = fwdret['simdata']

        field = np.array(inc_field)
        field = np.squeeze(field)
        field = np.flip(field)

        plt.figure()
        plt.imshow(field, cmap='gray')
        ax = plt.gca()
        ax.set_aspect(0.05)

        plt.show()


def main():
    test_model_parallel_forward_solver()

if __name__ == '__main__':
    main()
