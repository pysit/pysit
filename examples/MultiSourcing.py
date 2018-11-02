# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import horizontal_reflector

if __name__ == '__main__':
    #    Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    m = CartesianMesh(d, 91, 71)

    #    Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    Nshots = 50
    shots = []

    xmin = d.x.lbound
    xmax = d.x.rbound
    nx   = m.x.n
    zmin = d.z.lbound
    zmax = d.z.rbound

    source_list = []
    for i in range(Nshots):
        source_list.append(PointSource(m, (xmax*(i+1.0)/(Nshots+1.0), 0.1), RickerWavelet(10.0), intensity = (1))) #intensity of sources is different

    #2 PointSource objects are defined above. Group them together in a single SourceSet
    source_set = SourceSet(m,source_list)

    # Define set of receivers
    zpos = zmin + (1./9.)*zmax
    xpos = np.linspace(xmin, xmax, nx)
    receivers = ReceiverSet(m, [PointReceiver(m, (x, zpos)) for x in xpos])

    # Create and store the shot
    shot = Shot(source_set, receivers)




    # Define and configure the wave solver
    trange = (0.0,3.0)

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=6,
                                         trange=trange,
                                         kernel_implementation='cpp')
    true_model = solver.ModelParameters(m,{'C': C})
    modeling_tools = TemporalModeling(solver)
    ret = modeling_tools.forward_model(shot, true_model, return_parameters=['wavefield'])
    vis.animate(ret['wavefield'], m, display_rate=5)