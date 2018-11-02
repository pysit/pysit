import pytest

from pysit.core import PML, Dirichlet, RectangularDomain, CartesianMesh

from pysit.solvers import ConstantDensityAcousticWave
from pysit.solvers import ConstantDensityHelmholtz

from ..constant_density_acoustic import(ConstantDensityAcousticTimeScalar_1D_numpy,
                                        ConstantDensityAcousticTimeScalar_1D_cpp,
                                        ConstantDensityAcousticTimeScalar_2D_numpy,
                                        ConstantDensityAcousticTimeScalar_2D_cpp,
                                        ConstantDensityAcousticTimeScalar_3D_numpy,
                                        ConstantDensityAcousticTimeScalar_3D_cpp,
                                        ConstantDensityAcousticTimeODE_1D,
                                        ConstantDensityAcousticTimeODE_2D,
                                        ConstantDensityAcousticTimeODE_3D)

from ..constant_density_acoustic import(ConstantDensityAcousticFrequencyScalar_1D,
                                        ConstantDensityAcousticFrequencyScalar_2D,
                                        ConstantDensityAcousticFrequencyScalar_3D)

from pysit.util.basic_registration_factory import NoMatchError


class TestConstantDensityAcousticWaveFactory(object):

    def setup(self):
        pml = PML(0.1, 100)
        dirichlet = Dirichlet()

        x_config = (0.0, 1.0, pml, dirichlet)
        y_config = (0.0, 0.8, pml, dirichlet)
        z_config = (0.0, 0.6, dirichlet, pml)

        d1 = RectangularDomain(z_config)
        self.m1 = CartesianMesh(d1, 60)

        d2 = RectangularDomain(x_config, z_config)
        self.m2 = CartesianMesh(d2, 100, 60)

        d3 = RectangularDomain(x_config, y_config, z_config)
        self.m3 = CartesianMesh(d3, 100, 80, 60)

    def test_defaults(self):

        T = type(ConstantDensityAcousticWave(self.m1))
        assert T == ConstantDensityAcousticTimeScalar_1D_numpy

        T = type(ConstantDensityAcousticWave(self.m2))
        assert T == ConstantDensityAcousticTimeScalar_2D_numpy

        T = type(ConstantDensityAcousticWave(self.m3))
        assert T == ConstantDensityAcousticTimeScalar_3D_numpy

    def test_alternate_kernel(self):

        T = type(ConstantDensityAcousticWave(self.m1,
                                             kernel_implementation='cpp'))
        assert T == ConstantDensityAcousticTimeScalar_1D_cpp

        T = type(ConstantDensityAcousticWave(self.m2,
                                             kernel_implementation='cpp'))
        assert T == ConstantDensityAcousticTimeScalar_2D_cpp

        T = type(ConstantDensityAcousticWave(self.m3,
                                             kernel_implementation='cpp'))
        assert T == ConstantDensityAcousticTimeScalar_3D_cpp

        T = type(ConstantDensityAcousticWave(self.m1,
                                             kernel_implementation='omp'))
        assert T == ConstantDensityAcousticTimeScalar_1D_cpp

        T = type(ConstantDensityAcousticWave(self.m2,
                                             kernel_implementation='omp'))
        assert T == ConstantDensityAcousticTimeScalar_2D_cpp

        T = type(ConstantDensityAcousticWave(self.m3,
                                             kernel_implementation='omp'))
        assert T == ConstantDensityAcousticTimeScalar_3D_cpp


        with pytest.raises(NoMatchError):
            T = type(ConstantDensityAcousticWave(self.m1,
                                                 equation_formulation='ode',
                                                 kernel_implementation='cpp'))

    def test_spatial_accuracy(self):

        T = type(ConstantDensityAcousticWave(self.m1,
                                             kernel_implementation='numpy',
                                             spatial_accuracy_order=16))
        assert T == ConstantDensityAcousticTimeScalar_1D_numpy

        T = type(ConstantDensityAcousticWave(self.m1,
                                             kernel_implementation='cpp',
                                             spatial_accuracy_order=8))
        assert T == ConstantDensityAcousticTimeScalar_1D_cpp

        with pytest.raises(NoMatchError):
            T = type(ConstantDensityAcousticWave(self.m1,
                                                 equation_formulation='scalar',
                                                 kernel_implementation='cpp',
                                                 spatial_accuracy_order=16))

        with pytest.raises(NoMatchError):
            T = type(ConstantDensityAcousticWave(self.m1,
                                                 spatial_accuracy_order=3))

    def test_ode(self):


        S = ConstantDensityAcousticWave(self.m1, equation_formulation='ode')
        T = type(S)
        assert T == ConstantDensityAcousticTimeODE_1D
        assert S.temporal_accuracy_order == 4
        assert S.temporal_integrator == 'rk'


        T = type(ConstantDensityAcousticWave(self.m2,
                                             equation_formulation='ode'))
        assert T == ConstantDensityAcousticTimeODE_2D

        T = type(ConstantDensityAcousticWave(self.m3,
                                             equation_formulation='ode'))
        assert T == ConstantDensityAcousticTimeODE_3D

        T = type(ConstantDensityAcousticWave(self.m2,
                                             equation_formulation='ode',
                                             temporal_integrator='rk'))
        assert T == ConstantDensityAcousticTimeODE_2D

        S = ConstantDensityAcousticWave(self.m2,
                                        equation_formulation='ode',
                                        temporal_integrator='rk',
                                        temporal_accuracy_order=2)
        T = type(S)
        assert T == ConstantDensityAcousticTimeODE_2D
        assert S.temporal_accuracy_order == 2

        with pytest.raises(NoMatchError):
            T = type(ConstantDensityAcousticWave(self.m2,
                                                 equation_formulation='ode',
                                                 temporal_integrator='rk',
                                                 temporal_accuracy_order=6))

    def test_nonsupport_keywords(self):

        T = type(ConstantDensityAcousticWave(self.m2,
                                             not_a_support_keyword='random-text'))
        assert T == ConstantDensityAcousticTimeScalar_2D_numpy


class TestConstantDensityHelmholtz(object):

    def setup(self):
        pml = PML(0.1, 100)
        dirichlet = Dirichlet()

        x_config = (0.0, 1.0, pml, dirichlet)
        y_config = (0.0, 0.8, pml, dirichlet)
        z_config = (0.0, 0.6, dirichlet, pml)

        d1 = RectangularDomain(z_config)
        self.m1 = CartesianMesh(d1, 60)

        d2 = RectangularDomain(x_config, z_config)
        self.m2 = CartesianMesh(d2, 100, 60)

        d3 = RectangularDomain(x_config, y_config, z_config)
        self.m3 = CartesianMesh(d3, 100, 80, 60)

    def test_defaults(self):

        T = type(ConstantDensityHelmholtz(self.m1))
        assert T == ConstantDensityAcousticFrequencyScalar_1D

        T = type(ConstantDensityHelmholtz(self.m2))
        assert T == ConstantDensityAcousticFrequencyScalar_2D

        T = type(ConstantDensityHelmholtz(self.m3))
        assert T == ConstantDensityAcousticFrequencyScalar_3D

    def test_spatial_accuracy(self):

        S = ConstantDensityHelmholtz(self.m1, spatial_accuracy_order=16)
        T = type(S)
        assert T == ConstantDensityAcousticFrequencyScalar_1D
        assert S.spatial_accuracy_order == 16

        with pytest.raises(NoMatchError):
            T = type(ConstantDensityHelmholtz(self.m1, spatial_accuracy_order=3))

    def test_nonsupport_keywords(self):

        T = type(ConstantDensityHelmholtz(self.m2,
                                                 not_a_support_keyword='random-text'))
        assert T == ConstantDensityAcousticFrequencyScalar_2D
