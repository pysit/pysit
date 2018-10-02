# Std import block
import time

from pysit import *
from pysit.gallery import horizontal_reflector

def test_1d():
    pmlz = PML(0.1, 100, ftype='quadratic')

#   pmlz = Dirichlet()

    z_config = (0.1, 0.8, pmlz, Dirichlet())
    z_config = (0.1, 0.8, pmlz, pmlz)
#   z_config = (0.1, 0.8, Dirichlet(), Dirichlet())

    d = RectangularDomain(z_config)

    m = CartesianMesh(d, 301)

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    shots = []

    # Define source location and type
    zpos = 0.2
    source = PointSource(m, (zpos), RickerWavelet(25.0))

    # Define set of receivers
    receiver = PointReceiver(m, (zpos))
    # receivers = ReceiverSet([receiver])

    # Create and store the shot
    shot = Shot(source, receiver)
    # shot = Shot(source, receivers)
    shots.append(shot)


    # Define and configure the wave solver
    trange = (0.0, 3.0)

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=2,
                                         trange=trange,
                                         kernel_implementation='cpp')

    # Generate synthetic Seismic data
    tt = time.time()
    wavefields =  []
    base_model = solver.ModelParameters(m,{'C': C})
    generate_seismic_data(shots, solver, base_model, wavefields=wavefields)

    print('Data generation: {0}s'.format(time.time()-tt))

    return wavefields, m

def test_2d():
    # Setup

    #   Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    m = CartesianMesh(d, 90, 70)

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)


    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=1,
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
    base_model = solver.ModelParameters(m,{'C': C})
    generate_seismic_data(shots, solver, base_model, wavefields=wavefields)

    print('Data generation: {0}s'.format(time.time()-tt))

    return wavefields, m

def test_3d():
    # Setup

    #   Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 1.0, pmlx, pmlx)
    y_config = (0.1, 0.9, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, y_config, z_config)

    m = CartesianMesh(d, 90, 80, 70)

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)


    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=(1,1),
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   )

    # Define and configure the wave solver
    trange = (0.0,1.0)

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=2,
                                         trange=trange,
                                         kernel_implementation='cpp')

    # Generate synthetic Seismic data
    tt = time.time()
    wavefields =  []
    base_model = solver.ModelParameters(m,{'C': C})
    generate_seismic_data(shots, solver, base_model, wavefields=wavefields)

    print('Data generation: {0}s'.format(time.time()-tt))

    return wavefields, m

if __name__ == '__main__':

    from pysit import *

    # ws1, m1 = test_1d()
    # vis.animate(ws1, m1, display_rate=10)

    # ws2, m2 = test_2d()
    # vis.animate(ws2, m2, display_rate=10)

    ws3, m3 = test_3d()
    vis.animate(ws3, m3, display_rate=10)

