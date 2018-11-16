# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt
import math

import sys

from pysit import *
from pysit.gallery import horizontal_reflector

from pysit.util.parallel import *

from mpi4py import MPI

if __name__ == '__main__':
    # Setup

    comm = MPI.COMM_WORLD
#   comm = MPI.COMM_SELF
    size = comm.Get_size()
    rank = comm.Get_rank()

    pwrap = ParallelWrapShot(comm=MPI.COMM_WORLD)

    if rank == 0:
        ttt = time.time()

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

    Nshots = size
    sys.stdout.write("{0}: {1}\n".format(rank, Nshots // size))

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=Nshots,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   parallel_shot_wrap=pwrap,
                                   )


    # Define and configure the wave solver
    trange = (0.0,3.0)

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=6,
                                         trange=trange,
                                         kernel_implementation='cpp')

    # Generate synthetic Seismic data
    sys.stdout.write('Generating data...')
    base_model = solver.ModelParameters(m,{'C': C})
    tt = time.time()
    generate_seismic_data(shots, solver, base_model)
    sys.stdout.write('{1}:Data generation: {0}s\n'.format(time.time()-tt,rank))

    sys.stdout.flush()

    comm.Barrier()

    if rank == 0:
        tttt = time.time()-ttt
        sys.stdout.write('Total wall time: {0}\n'.format(tttt))
        sys.stdout.write('Total wall time/shot: {0}\n'.format(tttt/Nshots))

    objective = TemporalLeastSquares(solver, parallel_wrap_shot=pwrap)

    # Define the inversion algorithm
#   invalg = GradientDescent(objective)
    invalg = LBFGS(objective, memory_length=10)
    initial_value = solver.ModelParameters(m, {'C': C0})

    # Execute inversion algorithm
    print('Running LBFGS...')
    tt = time.time()

    nsteps = 2

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


    line_search = 'backtrack'

    result = invalg(shots, initial_value, nsteps,
                    line_search=line_search,
                    status_configuration=status_configuration, verbose=True)

    print('Run time:  {0}s'.format(time.time()-tt))


    if rank == 0:

        model = result.C.reshape(m.shape(as_grid=True))

        from scipy.io import savemat

        out = {'result':model, 'true':C.reshape(m.shape(as_grid=True))}
        savemat('test.mat',out)


    # Do something to visualize the results
#   display_on_grid(C, d, shade_pml=True)
#   display_on_grid(result.C, d, shade_pml=True)
    #display_seismogram(shots[0], clim=[-1,1])
    #display_seismogram(shots[0], wiggle=True, wiggle_skip=1)
    # animate_wave_evolution(ps, domain=d, display_rate=10, shade_pml=True)