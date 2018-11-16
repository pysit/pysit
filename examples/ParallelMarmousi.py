import time
import sys

import numpy as np

from pysit import *
from pysit.gallery import marmousi

from pysit.util.parallel import ParallelWrapShot

from mpi4py import MPI

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Wrapper for handling parallelism over shots
    pwrap = ParallelWrapShot(comm=MPI.COMM_WORLD)

    if rank == 0:
        ttt = time.time()

    #   Load or generate true wave speed
    C, C0, m, d = marmousi(patch='mini_square')

    # Set up shots

    Nshots = size
    sys.stdout.write("{0}: {1}\n".format(rank, Nshots // size))

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=Nshots,
                                   source_depth=20.0,
                                   receivers='max',
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
    data_gen_time = time.time()-tt
    sys.stdout.write('{1}:Data generation: {0}s\n'.format(data_gen_time,rank))
    sys.stdout.flush()

    comm.Barrier()

    if rank == 0:
        tttt = time.time()-ttt
        sys.stdout.write('Total wall time: {0}\n'.format(tttt))
        sys.stdout.write('Total wall time/shot: {0}\n'.format(tttt/Nshots))

    objective = TemporalLeastSquares(solver,
                                     parallel_wrap_shot=pwrap)

    # Define the inversion algorithm
    invalg = LBFGS(objective, memory_length=10)

    initial_value = solver.ModelParameters(m, {'C': C0})

    # Execute inversion algorithm
    print('Running LBFGS...')
    tt = time.time()

    nsteps = 10

    status_configuration = {'value_frequency'           : 1,
                            'objective_frequency'       : 1}

    result = invalg(shots, initial_value, nsteps,
                    line_search='backtrack',
                    status_configuration=status_configuration,
                    verbose=True)

    print('Run time:  {0}s'.format(time.time()-tt))


    if rank == 0:

        model = result.C.reshape(m.shape(as_grid=True))

        vals = list()
        for k,v in list(invalg.objective_history.items()):
            vals.append(v)
        obj_vals = np.array(vals)

        from scipy.io import savemat

        out = {'result': model,
               'true': C.reshape(m.shape(as_grid=True)),
               'initial': C0.reshape(m.shape(as_grid=True)),
               'obj': obj_vals
              }
        savemat('marm_recon.mat',out)
