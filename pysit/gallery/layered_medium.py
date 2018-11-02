

import copy
import tarfile
import os
import os.path
import itertools
import math

import numpy as np
import scipy.signal as signal

from pysit.util.image_processing import blur_image
from pysit.gallery.gallery_base import GeneratedGalleryModel

from pysit import * #PML, Domain


__all__ = ['LayeredMediumModel', 'layered_medium', 'Layer']


class Layer(object):
    def __init__(self, velocity, thickness, label=None, fixed=False):

        self.velocity = velocity
        self.thickness = thickness
        self.label = label
        self.fixed = fixed

_air   = Layer(300.0, 30, 'air', fixed=True)
_water = Layer(1500.0, 120, 'water', fixed=True)
_rock  = Layer(3200.0, 1800, 'rock')

water_rock = [_water, _rock]

_rock_velocitys = [-288.0, -150.0, -36.0, -18.0, -90.0, 360.0, -60.0, -450.0, 0.0, 78.0]
water_layered_rock = [_water] + [Layer(s+3300, 180, 'rock {0}'.format(i)) for s,i in zip(_rock_velocitys, itertools.count())]


class LayeredMediumModel(GeneratedGalleryModel):

    """ Gallery model for a generic, flat, layered medium. """

    model_name =  "Layered"

    valid_dimensions = (1,2,3)

    @property #read only
    def dimension(self):
        return self.domain.dim

    supported_physics = ('acoustic',)

    @property
    def z_length(self):
        return float(sum([L.thickness for L in self.layers]))

    def __init__(self, layers,
                       z_delta=None,
                       min_ppw_at_freq=(6,10.0), # 6ppw at 10hz
                       x_length=None, x_delta=None,
                       y_length=None, y_delta=None,
                       initial_model_style='smooth',
                       initial_config={'sigma':100.0, 'filtersize':100},
                       **kwargs):
        """ Constructor for a constant background model with horizontal reflectors.

        Parameters
        ----------
        layers : list
            List of Layer objects
        z_delta : float, optional
            Minimum mesh spacing in depth direction, see Notes.
        min_ppw_at_freq : tuple (int, float)
            Tuple with structure (min_ppw, peak_freq) to set the minimum points-per-wavelength at the given peak frequency.
        x_length : float
            Physical size in x direction
        x_delta : float
            Grid spacing in x direction
        y_length : float
            Physical size in y direction
        y_delta : float
            Grid spacing in y direction
        initial_model_style : {'smooth', 'constant', 'gradient'}
            Setup for the initial model.
        initial_config : dict
            Configuration parameters for initial models.

        Notes
        -----
        * If z_delta is not set, min_ppw_at_freq is used.  z_delta overrides use
        of min_ppw_at_freq.

        * Domain will be covered exactly, so z_delta is the maximum delta, it
        might actually end up being smaller, as the delta is determined by the
        mesh class.

        """

        GeneratedGalleryModel.__init__(self)

        self.layers = layers

        self.min_z_delta = z_delta
        self.min_ppw_at_freq = min_ppw_at_freq

        self.x_length = x_length
        self.x_delta = x_delta
        self.y_length = y_length
        self.y_delta = y_delta

        self.initial_model_style = initial_model_style
        self.initial_config = initial_config

        # Set _domain and _mesh
        self.build_domain_and_mesh(**kwargs)

        # Set _initial_model and _true_model
        self.rebuild_models()

    def build_domain_and_mesh(self, **kwargs):
        """ Constructs a mesh and domain for layered media. """

        # Compute the total depth
        z_length = self.z_length

        x_length = self.x_length
        x_delta = self.x_delta
        y_length = self.y_length
        y_delta = self.y_delta

        # If the minimum z delta is not specified.
        if self.min_z_delta is None: #use min_ppw & peak_frequency
            min_ppw, peak_freq = self.min_ppw_at_freq
            min_velocity = min([L.velocity for L in self.layers])
            wavelength = min_velocity / peak_freq
            z_delta = wavelength / min_ppw
        else:
            z_delta = self.min_z_delta

        z_points = math.ceil(z_length/z_delta)

        # Set defualt z boundary conditions
        z_lbc = kwargs['z_lbc'] if ('z_lbc' in list(kwargs.keys())) else PML(0.1*z_length, 100.0)
        z_rbc = kwargs['z_rbc'] if ('z_rbc' in list(kwargs.keys())) else PML(0.1*z_length, 100.0)

        domain_configs = list()
        mesh_args = list()

        # If a size of the x direction is specified, determine those parameters
        if x_length is not None:
            if x_delta is None:
                x_delta = z_delta
            x_points = math.ceil(float(x_length)/x_delta)

            # Set defualt x boundary conditions
            x_lbc = kwargs['x_lbc'] if ('x_lbc' in list(kwargs.keys())) else PML(0.1*x_length, 100.0)
            x_rbc = kwargs['x_rbc'] if ('x_rbc' in list(kwargs.keys())) else PML(0.1*x_length, 100.0)

            domain_configs.append((0, x_length, x_lbc, x_rbc))
            mesh_args.append(x_points)

            # the y dimension only exists for 3D proble, so only if x is defined
            if y_length is not None:
                if y_delta is None:
                    y_delta = z_delta
                y_points = math.ceil(float(y_length)/y_delta)

                # Set defualt y boundary conditions
                y_lbc = kwargs['y_lbc'] if ('y_lbc' in list(kwargs.keys())) else PML(0.1*y_length, 100.0)
                y_rbc = kwargs['y_rbc'] if ('y_rbc' in list(kwargs.keys())) else PML(0.1*y_length, 100.0)

                domain_configs.append((0, y_length, y_lbc, y_rbc))
                mesh_args.append(y_points)

        domain_configs.append((0, z_length, z_lbc, z_rbc))
        mesh_args.append(z_points)

        self._domain = RectangularDomain(*domain_configs)

        # Build mesh
        mesh_args = [self._domain] + mesh_args
        self._mesh = CartesianMesh(*mesh_args)

    def rebuild_models(self):
        """ Rebuild the true and initial models based on the current configuration."""

        sh = self._mesh.shape(as_grid=True)
        _shape_tuple = tuple([1]*(len(sh)-1) + [sh[-1]]) # ones in each dimension except for Z
        _pad_tuple = [(0,n-1) for n in sh]
        _pad_tuple[-1] = (0,0)
        _pad_tuple = tuple(_pad_tuple)

        # Construct true velocity profile
        vp = np.zeros(_shape_tuple)
        grid = self._mesh.mesh_coords(sparse=True)
        ZZ = grid[-1].reshape(_shape_tuple)

        total_filled = 0
        for L in self.layers[::-1]:
            cutoff_depth = self.z_length - total_filled
            vp[ZZ <= cutoff_depth] = L.velocity

            total_filled += L.thickness

        # Construct initial velocity profile:
        if self.initial_model_style == 'constant': # initial_config = {'velocity': 3000.0}
            vp0 = np.ones(_shape_tuple)*self.initial_config['velocity']

        elif self.initial_model_style == 'smooth': #initial_config = {'sigma':50.0, 'filtersize':8}
            vp0 = blur_image(vp.reshape(-1,),
                             self.initial_config['filtersize'],
                             self.initial_config['sigma'],
                             mesh_deltas=(self._mesh.z.delta,))
            vp0.shape = vp.shape

        elif self.initial_model_style == 'gradient': # initial_config = {}
            # collect the non-fixed layers for choosing the gradient bounds
            velocities = [L.velocity for L in self.layers if not L.fixed]
            cutoff_depth = 0

            # find the first non-fixed layer to start the gradient at.
            for L in self.layers:
                if L.fixed:
                    cutoff_depth += L.thickness
                else:
                    break
            vp0 = vp.copy()
            loc = np.where(ZZ > cutoff_depth)
            vp0[loc] = np.linspace(min(velocities), max(velocities), loc[0].size)

        # Fix the fixed layers
        old_depth = 0
        for L in self.layers:
            depth = old_depth + L.thickness
            if L.fixed:
                vp0[(ZZ >= old_depth) & (ZZ < depth)] = L.velocity
            old_depth = depth


        # Construct final padded velocity profiles
        C = np.pad(vp, _pad_tuple, 'edge').reshape(self._mesh.shape())
        C0 = np.pad(vp0, _pad_tuple, 'edge').reshape(self._mesh.shape())
        self._true_model = C
        self._initial_model = C0

def layered_medium( layers=water_layered_rock, **kwargs):
    """ Friendly wrapper for instantiating the layered medium model. """

    # Setup the defaults
    model_config = dict(z_delta=None,
                        min_ppw_at_freq=(6,10.0), # 6ppw at 10hz
                        x_length=None, x_delta=None,
                        y_length=None, y_delta=None,
                        initial_model_style='smooth',
                        initial_config={'sigma':100.0, 'filtersize':100})

    # Make any changes
    model_config.update(kwargs)

    return LayeredMediumModel(layers, **model_config).get_setup()

#if __name__ == '__main__':
#
##  ASD = LayeredMediumModel(water_layered_rock)
##  ASD = LayeredMediumModel(water_layered_rock, initial_model_style='smooth', initial_config={'sigma':100, 'filtersize':150})
##  ASD = LayeredMediumModel(water_layered_rock, initial_model_style='gradient')
##  ASD = LayeredMediumModel(water_layered_rock, initial_model_style='constant', initial_config={'velocity':3000})
##  ASD = LayeredMediumModel(water_layered_rock, x_length=2000.0, y_length=1000.0)
#
#   C, C0, m, d = layered_medium(x_length=2000)
#
#   import matplotlib.pyplot as plt
#
#   fig = plt.figure()
#   fig.add_subplot(2,1,1)
#   vis.plot(C, m)
#   fig.add_subplot(2,1,2)
#   vis.plot(C0, m)
#   plt.show()
