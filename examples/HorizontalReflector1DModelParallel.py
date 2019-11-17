import time

import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from pysit import *
from pysit.core.mesh import ParallelCartesianMesh
from pysit.modeling.temporal_modeling import TemporalModeling
from pysit.gallery import horizontal_reflector

def display_decomposition(p):

    data = {
        'len_unpadded' : p.domain_local.z.length,
        'lbound_unpadded' : p.domain_local.z.lbound,
        'rbound_unpadded' : p.domain_local.z.rbound,
        'len_padded' : p.domain_local.z.length + p.domain_local.z.lbc.length \
            + p.domain_local.z.rbc.length,
        'lbound_padded' : p.domain_local.z.lbound - p.domain_local.z.lbc.length,
        'rbound_padded' : p.domain_local.z.rbound + p.domain_local.z.rbc.length,
    }

    data = p.comm.gather(data, root=0)

    if p.comm.rank == 0:
        
        ax = plt.gca()
        
        for i, d in enumerate(data):

            color = 'r'

            ax.axvline(x=d['lbound_unpadded'], color=color)
            ax.axvline(x=d['rbound_unpadded'], color=color)
            ax.axvline(x=d['lbound_padded'], color=color, linestyle='--')
            ax.axvline(x=d['rbound_padded'], color=color, linestyle='--')
            

    else:
        assert data is None

def plot_space_time(us, title=None):

    ax = plt.gca()

    # Imshow on array of calculated values
    arr = np.array(us)
    arr = arr.reshape(arr.shape[0], arr.shape[1])
    ax.imshow(arr, cmap='gray')

    # Plot labels
    ax.set_xlabel('space')
    ax.set_ylabel('time')
    ax.set_title(title)

    ax.set_ylim(0, 3.0)
    ax.set_aspect(20)

if __name__ == '__main__':
   
    solver_accuracy_order = 2
    solver_padding = solver_accuracy_order // 2

    # Define global domain
    pmlz = PML(0.1, 100, ftype='quadratic')
    z_config = (0.1, 0.7, pmlz, pmlz)
    d = RectangularDomain(z_config)

    # Parallel mesh
    m = ParallelCartesianMesh(d, solver_padding, MPI.COMM_WORLD, 101)
  
    print(f'Rank {m.rank} has mesh delta = {m.mesh_local.z.delta}')

    # Plot domain decomp
    fig = plt.figure()
    display_decomposition(m) 
    if m.rank == 0:
        plt.show()
