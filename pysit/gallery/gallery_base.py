import os
import os.path
import urllib.request, urllib.error, urllib.parse
import urllib.parse
import hashlib
import gzip

import numpy as np
import scipy

from pysit import *
from pysit.util.config import get_gallery_data_path
from pysit.util.net import download_file
from pysit.util.io import read_model
from pysit.util.image_processing import blur_image

__all__ = ['GalleryModel', 'GeneratedGalleryModel', 'PrecomputedGalleryModel']

class GalleryModel(object):
    """ Base class for gallery models in PySIT.

    Attributes
    ----------
    model_name : string, class attribute
        Long form name of the model.
    valid_dimensions : iterable, class attribute
        List of the dimensions for which the model is valid.
    supported_physics : list, class attribute
        List of the physics (e.g., acoustic, or elastic) supported by the model.
    physics_to_parameters_map : dict, class attribute
        Dict which maps physics to the list of parameters needed for that physics.
    dimension : int
        Dimension of the actual instance of the gallery model.
    true_model : ndarray
        The true model for a gallery instance.
    initial_model : ndarray
        The initial model for a gallery instance.
    mesh : pysit.mesh
        The computational mesh on which the model is defined.
    domain : pysit.domain
        The physical domain on which the model is defined.

    """


    model_name = ""

    valid_dimensions = tuple()

    # List of physics supported for this gallery problem
    supported_physics = list()

    physics_to_parameters_map = {'acoustic' : ['vp'],
                                 'variable-density-acoustic' : ['vp', 'density'],
                                 'elastic' : ['vp', 'vs', 'density']}

    @property #read only
    def dimension(self):
        raise NotImplementedError('dimension must be set in subclass')

    def __init__(self, *args, **kwargs):
        self._true_model = None
        self._initial_model = None
        self._mesh = None
        self._domain = None

    def get_setup(self):
        """ Return the usual 4-tuple of (true_model, initial_model, mesh, and
            domain) needed for an experiment."""

        true_model = self.true_model

        initial_model = self.initial_model

        mesh = self.mesh

        domain = self.domain

        return true_model, initial_model, mesh, domain

    @property
    def true_model(self):
        return self._true_model

    @property
    def initial_model(self):
        return self._initial_model

    @property
    def mesh(self):
        return self._mesh

    @property
    def domain(self):
        return self._domain

class GeneratedGalleryModel(GalleryModel):
    """ Empty base class for generated gallery models."""
    pass



class PrecomputedGalleryModel(GalleryModel):
    """ Base class for pre-computed or community gallery models in PySIT.

    Precomputed models are licensed (or open) community models.  Given that
    distribution licensing is tricky, this class is designed to download any
    model that is not already downloaded to the location specified in the pysit
    configuration file.  In this manner, the user is responsible for ensuring
    that they have the proper approval to use the community model.

    Any rescalings and crops are cached in the user's specified cache folder to
    speed up computation.

    Attributes
    ----------
    fs_full_name : string, class attribute
        Longer name for use in file system calls.
    fs_short_name : string, class attribute
        Short name for use in file system calls.
    supported_physical_parameters : iterable, class attribute
        List of the physical parameters supported by the gallery model.
    supported_masks : iterable, class attribute
        List of the masks supported by the gallery model.
    base_physical_origin : iterable, class attribute
        Origin of the full sized model coordinate system.
    base_physical_size : iterable, class attribute
        Size (in physical units) of the full sized model coordinate system.
    base_physical_dimensions_units : list, class attribute
        String with the spatial units for the coordinate system.
    base_pixels : iterable, class attribute
        Number of pixels in the full sized model.
    base_pixel_scale : iterable, class attribute
        Size of the pixels in the full sized model.
    base_pixel_units : list, class attribute
        Units for the the pixels in the full sized model.
    water_properties : tuple, class attribute
        Specifies how water layers are determined in the model:
            * (None, ) indicates no water
            * ('depth', <depth: float>) specifies that a preset depth from the base_physical_origin is to be used
            * ('mask', ) indicates that the gallery model's water mask should be used
    patches : dict, class attribute
        Mapping of pre-specified useful patches of the gallery model to their origin and sizes.

    Methods
    -------
    rebuild_models :
        Rebuilds true and initial models if changes are made to the underlying specification.

    """

    # A sanitized name for filesystem work
    fs_full_name = ""
    fs_short_name = ""

    # Available data
    supported_physical_parameters = list()  # ['density', 'vp']
    supported_masks = list()  # ['salt', 'water']

    # The local file names used to save the downloaded models
    _local_parameter_filenames = {}  # {'vp' : None, 'vs' : None, 'density' : None}
    _local_mask_filenames = {}  # {'water-mask' : None, 'salt-mask' : None}

    # Sometimes the sources are not in the units we expect.  This scale factor
    # allows the numbers to be converted into common units.
    _parameter_scale_factor = {}  # { 'vp' : 1.0, 'density' : 1.0}

    # Dictionary mapping physical parameters to a list of URLs where they
    # can be obtained
    _remote_file_sources = {}

    # Sometimes the model data is stored transposed from how pysit expects it.
    _model_transposed = False

    # Model specification
    base_physical_origin = None
    base_physical_size = None
    base_physical_dimensions_units = tuple()

    base_pixels = None
    base_pixel_scale = None
    base_pixel_units = tuple()

    # Water properties specify the way the water masking is handled
    water_properties = (None, )

    # Default configuration of the four possible initial model schemes.
    _initial_configs = {'smooth_width': {'sigma': 3000.0},
                        'smooth_low_pass': {'freq': 1/3000.},
                        'constant': {'velocity': 3000.0},
                        'gradient': {'min': 1500.0, 'max': 3000}}

    # Maps names of scales to physical scales for the model
    _scale_map = {}

    # Common array patches
    patches = {}

    # Short cuts for the left and right positions of the edges of the domain.
    @property
    def _coordinate_lbounds(self):
        return self.physical_origin

    @property
    def _coordinate_rbounds(self):
        return self.physical_origin + self.physical_size

    license_string = None

    def __init__(self, physics='acoustic',
                       origin=None,
                       size=None,
                       pixel_scale='mini',
                       pixels=None,
                       initial_model_style='smooth_low_pass',
                       initial_config={},
                       fix_water_layer=True,
                       **kwargs):

        """ Constructor for community gallery models.

        Parameters
        ----------
        physics : {'acoustic', 'variable-density-acoustic', 'elastic'}
            Wave physics to instantiate.
        origin : array-like, optional
            Origin, in physical coordinates, of the model patch.  Defaults to base origin.
        size : array-like, optional
            Size, in physical units, of the model patch.  Defaults to base size.
        pixel_scale : string or array-like
            Either key to pixel scale map, or array-like container of grid spacings
        pixels : array-like
            Number of pixels. This overrides pixel_scale.
        initial_model_style: {'smooth_low_pass', 'smooth_width', 'constant', 'gradient'}
            Style of the initial model, see note for more details.
        initial_config : dict
            Configuration of the initial model style
        fix_water_layer : boolean
            If true, water layer in initial model is set to match water layer in true model.
        kwargs : dict
            Additional keyword arguments for specifying boundary conditions and more.


        Notes
        -----
        The four initial model styles are:
        1) 'smooth_low_pass' -- Smooths the true model with a Gaussian low pass filter with specified cutoff frequency.
            Options: 'freq' : spatial frequency for the cutoff
        2) 'smooth_width' -- Smooths the true model with a Gaussian low pass filter with specified standard deviation.
            Options: 'sigma' : width of the smoothing filter
        3) 'constant' -- Initial value is a specified constant
            Options: 'velocity' : the velocity to use
        4) 'gradient' -- Sets initial velocity to a linear gradient
            Options: 'min' : Minimum value for the gradient.
                     'max' : Maximum value for the gradient.

        """

        # Validate physics and collect the names of the associated parameters.
        if physics not in self.supported_physics:
            raise ValueError('Physics \'{0}\' not supported by model {1}'.format(physics, self.__class__.__name__))

        # Currently only support accoustic physics.
        if physics != 'acoustic':
            raise NotImplementedError('Currently only support accoustic physics.')

        self.physics = physics

        # If origin is not set, use the model base.  Also verify that it is within valid bounds.
        if origin is not None:
            if np.all(origin >= self.base_physical_origin) and np.all(origin < (self.base_physical_size - self.base_physical_origin)):
                self.physical_origin = np.asarray(origin).reshape(2,)
            else:
                raise ValueError('Specified origin is outside of the model bounds.')
        else:
            self.physical_origin = self.base_physical_origin.copy()

        # If size is not set, use the model base.  Also verify that it is within valid bounds.
        if size is not None:
            size = np.asarray(size).reshape(2,)
            if np.all(size > 0.0) and np.all((size + self.physical_origin) <= self.base_physical_size):
                self.physical_size = size
            else:
                raise ValueError('Specified size is outside of the model bounds.')
        else:
            self.physical_size = self.base_physical_size.copy()

        # Setting the number of pixels trumps the pixel_scale
        if pixels is not None:
            self.pixels = np.asarray(pixels).reshape(2,)
            self.pixel_scale = self.physical_size / pixels

        elif pixel_scale is not None:
            if isinstance(pixel_scale, str):
                pixel_scale = self._scale_map[pixel_scale]

            self.pixel_scale = np.asarray(pixel_scale).reshape(2,)

            length = self._coordinate_rbounds - self._coordinate_lbounds
            self.pixels = np.ceil(length/self.pixel_scale).astype(np.int)

            # Correct physical size to match the number of pixels. This can go
            # over the true max size by up to 1 delta.
            self.physical_size = self.pixel_scale*self.pixels

            # Add 1 because the last point _is_ included.  Adds to all
            # dimensions.
            self.pixels += 1

        else:  # both are None
            self.pixels = self.base_pixels.copy()
            self.pixel_scale = self.base_pixel_scale.copy()

        # Create pysit domains and meshes

        # extract or default the boundary conditions
        x_lbc = kwargs['x_lbc'] if ('x_lbc' in kwargs) else PML(0.1*self.physical_size[0], 100.0)
        x_rbc = kwargs['x_rbc'] if ('x_rbc' in kwargs) else PML(0.1*self.physical_size[0], 100.0)
        z_lbc = kwargs['z_lbc'] if ('z_lbc' in kwargs) else PML(0.1*self.physical_size[1], 100.0)
        z_rbc = kwargs['z_rbc'] if ('z_rbc' in kwargs) else PML(0.1*self.physical_size[1], 100.0)

        x_config = (self._coordinate_lbounds[0], self._coordinate_rbounds[0], x_lbc, x_rbc)
        z_config = (self._coordinate_lbounds[1], self._coordinate_rbounds[1], z_lbc, z_rbc)
        d = RectangularDomain(x_config, z_config)

        mesh = CartesianMesh(d, *self.pixels)

        # store initial model setup values
        self.initial_model_style = initial_model_style
        self.initial_config = initial_config
        self.fix_water_layer = fix_water_layer

        # Store the mesh and domain, and build the model parameters
        self._mesh = mesh
        self._domain = mesh.domain
        self.rebuild_models()

    def rebuild_models(self):
        """ Rebuild the true and initial models based on the current configuration."""

        # Get the list of parameters required for the current physics
        parameter_names = self.physics_to_parameters_map[self.physics]

        # Get a list of properly scaled and cropped parameters arrays for each of those parameters
        parameter_arrays = [self._load_scaled_parameter_array(p) for p in parameter_names]

        # Velocity only, at the moment
        # Eventually all of this will need to be done for each model parameter
        vp = parameter_arrays[0]

        # Extract the depth coordinates of the grid points
        grid = self._mesh.mesh_coords()
        ZZ = grid[-1].reshape(vp.shape)

        # Copy the default values for the initial model style
        config = dict(self._initial_configs[self.initial_model_style])
        # Override them with user provided values
        config.update(self.initial_config)

        # Construct initial velocity profile:
        if self.initial_model_style == 'constant':
            vp0 = np.ones(vp.shape)*config['velocity']

        elif self.initial_model_style == 'smooth_width':
            vp0 = blur_image(vp,
                             sigma=config['sigma'],
                             mesh_deltas=self._mesh.deltas)

        elif self.initial_model_style == 'smooth_low_pass':
            vp0 = blur_image(vp,
                             freq=config['freq'],
                             mesh_deltas=self._mesh.deltas)

        elif self.initial_model_style == 'gradient':
            # construct the N-D depth trace shape, which is 1 in every dimension
            # except the depth direction
            sh = self._mesh.shape(as_grid=True)
            _shape_tuple = tuple([1]*(len(sh)-1) + [sh[-1]])

            # Construct the shape of the padding, which is 0 padding in the depth
            # direction and n pixels in each other direction
            _pad_tuple = [(0, n-1) for n in sh]
            _pad_tuple[-1] = (0, 0)
            _pad_tuple = tuple(_pad_tuple)

            # If the water is to be fixed, the gradient is started at the
            # shallowest water-rock interaction. Otherwise, start at the top.
            if self.fix_water_layer and (self.water_properties[0] is not None):

                grid_sparse = self._mesh.mesh_coords(sparse=True)
                ZZZ = grid_sparse[-1].reshape(_shape_tuple)

                min_max_depth = self._find_min_max_water_depth()

                loc = np.where(ZZZ > min_max_depth)
                zprof = np.zeros(_shape_tuple)
                zprof[loc] = np.linspace(config['min'], config['max'], loc[0].size)
            else:
                zprof = np.linspace(config['min'], config['max'], self._mesh.z.n)
                zprof.shape = _shape_tuple

            # Pad out the gradient.
            vp0 = np.pad(zprof, _pad_tuple, 'edge')

        if self.fix_water_layer and (self.water_properties[0] is not None):

            # If the water layer is specified by depth, reset everything shallower
            if self.water_properties[0] == 'depth':
                loc = np.where(ZZ < self._find_min_max_water_depth())
                vp0[loc] = vp[loc]

            # If the water mask is set, reset the masked values.
            elif self.water_properties[0] == 'mask':
                mask = self._load_scaled_parameter_array('water')
                self._water_mask = mask
                loc = np.where(mask != 0.0)
                vp0[loc] = vp[loc]

        # Construct final padded velocity profiles
        C = vp.reshape(self._mesh.shape())
        C0 = vp0.reshape(self._mesh.shape())

        self._true_model = C
        self._initial_model = C0

    def _find_min_max_water_depth(self):
        """ Returns the shallowest water-rock interation depth. """

        if self.water_properties[0] == 'depth':
            return self.water_properties[1]

        elif self.water_properties[0] == 'mask':
            mask = self._load_scaled_parameter_array('water')

            grid = self._mesh.mesh_coords()
            ZZ = grid[-1].reshape(mask.shape)

            return np.min((ZZ*mask).max(axis=-1))
        else:
            return 0.0

    @classmethod
    def _get_gallery_model_path(cls):
        """ Returns the full path of the gallery data location."""
        gdp = get_gallery_data_path()
        path = os.path.join(gdp, cls.fs_full_name)

        if not os.path.exists(path):
            os.mkdir(path)
        return path

    @classmethod
    def _build_local_parameter_path(cls, param):
        """ Builds the full path to the local parameter or mask file."""

        file_path = cls._get_gallery_model_path()

        if param in cls.supported_physical_parameters:
            parameter_fname = cls._local_parameter_filenames[param]
        elif param in cls.supported_masks:
            parameter_fname = cls._local_mask_filenames[param]

        return os.path.join(file_path, parameter_fname)

    @classmethod
    def _build_parameter_patch_filename(cls, param, hash_args, ext=''):
        """ Builds filename for rescaled model parameters of the form <fs_short_name>_<md5_str>.<ext>."""

        hash_str = hashlib.md5(b''.join([a.tostring() for a in hash_args])).hexdigest()

        return "{0}_{1}_{2}{3}".format(cls.fs_short_name, param, hash_str, ext)

    @classmethod
    def _build_parameter_patch_path(cls, param, hash_args, ext=''):
        """ Builds the full path to the local patch file."""

        file_path = cls._get_gallery_model_path()

        parameter_fname = cls._build_parameter_patch_filename(param, hash_args, ext)

        return os.path.join(file_path, parameter_fname)

    @classmethod
    def _write_npy(cls, arr, fname, gz=True):
        """ Writes the array as a numpy file (optionally gzipped)."""

        if fname.endswith('.gz'):
            f = fname[:-3]
            np.save(f, arr)
            with open(f, 'rb') as raw:
                with gzip.open(fname, 'wb') as gzz:
                    gzz.write(raw.read())
            os.remove(f)
        elif gz:
            np.save(fname, arr)
            with open(fname, 'rb') as raw:
                with gzip.open(fname.join('.gz'), 'wb') as gzz:
                    gzz.write(raw.read())
            os.remove(f)
        else:
            np.save(fname, arr)

    @classmethod
    def _read_npy(cls, fname):
        """ Reads a potentially gzipped numpy array file.  Gets around any
        inconsistencies in the numpy file writing specification."""

        if fname.endswith('.gz'):
            f = fname[:-3]
            with gzip.open(fname, 'rb') as gzz:
                arr = np.load(gzz)
        else:
            arr = np.load(fname)

        return arr

    @classmethod
    def _download_and_prepare(cls, param):
        """ Downloads and prepares parameter file for use.  Prepare means it reads the segy file and saves it as a gzipped numpy file."""

        # verify that the requested parameter is supported by this gallery model
        if param not in cls.supported_physical_parameters + cls.supported_masks:
            raise ValueError('{0} is invalid physical parameter/mask for gallery model {1}.'.format(param, cls.__name__))

        parameter_path = cls._build_local_parameter_path(param)

        # Check if file exists locally
        if not os.path.isfile(parameter_path):

            # If it does not, try downloading it from sources
            downloaded = False

            if cls.license_string is not None:

                print(cls.license_string)
                agree = input("Continue downloading {0} model '{1}' (yes/[no])?  ".format(cls.model_name, param) ) or 'no'
                if agree.strip().lower() != 'yes':
                    raise Exception('Cannot proceed without license agreement.')

            for url in cls._remote_file_sources[param]:

                try:
                    download_file(url, parameter_path)
                    downloaded = True
                except urllib.error.HTTPError:
                    continue

            # If no source is successful, raise an exception
            if not downloaded:
                raise Exception('Unable to download parameter.')

        # Now check and see if the 'full sized' patch version exists.  If it does not, build that.
        hash_args = (cls.base_physical_origin, cls.base_physical_size, cls.base_pixel_scale, cls.base_pixels)
        scaled_parameter_path = cls._build_parameter_patch_path(param, hash_args, '.npy.gz')

        # Check if file exists locally
        if not os.path.isfile(scaled_parameter_path):
            # create it by loading the segy file and saving

            # If the parameter file is gzipped, unzip it
            unpacked = False
            if parameter_path[-2:] == 'gz':
                unpacked_parameter_path = parameter_path[:-3]

                with gzip.open(parameter_path, 'rb') as infile:
                    with open(unpacked_parameter_path, 'wb') as outfile:
                        outfile.write(infile.read())

                parameter_path = unpacked_parameter_path
                unpacked = True

            arr = read_model(parameter_path)

            # Masks are not scaled
            if param not in cls.supported_masks:
                arr *= cls._parameter_scale_factor[param]

            if cls._model_transposed:
                arr = arr.T

            # Write it
            cls._write_npy(arr, scaled_parameter_path)

            if unpacked:
                # delete the unzipped version
                os.remove(parameter_path)

    def _resample_parameter(self, param):
        """ Resamples the specified parameter to the sizes, from the original."""

        # Load the original parameter array
        base_hash_args = (self.base_physical_origin, self.base_physical_size, self.base_pixel_scale, self.base_pixels)
        base_scaled_parameter_path = self._build_parameter_patch_path(param, base_hash_args, '.npy.gz')
        big_arr = self._read_npy(base_scaled_parameter_path)

        # Resample the array, in each dimension, with nearest neighbor interpolation.
        y = big_arr
        for i in range(big_arr.ndim):

            x = self.base_physical_origin[i] + np.arange(self.base_pixels[i])*self.base_pixel_scale[i]
            I = scipy.interpolate.interp1d(x, y, copy=False, axis=i, kind='nearest')

            new_x = self.physical_origin[i] + np.arange(self.pixels[i])*self.pixel_scale[i]

            # Bounds that go slightly past the true thing should extend from the right side
            new_x[new_x > self.base_physical_size[i]] = self.base_physical_size[i]

            y = I(new_x)

        return y

    def _load_scaled_parameter_array(self, param):
        """ Builds or loads a numpy array for the scaled version of the model file."""

        # Download and prep the file, if it is needed
        self._download_and_prepare(param)

        # Build scaled filename and its path
        hash_args = (self.physical_origin, self.physical_size, self.pixel_scale, self.pixels)
        scaled_parameter_path = self._build_parameter_patch_path(param, hash_args, '.npy.gz')

        # if file exists, load it
        if os.path.isfile(scaled_parameter_path):
            arr = self._read_npy(scaled_parameter_path)
        else:
            # otherwise we need to write that particular patch, as loaded from the default array.
            arr = self._resample_parameter(param)
            self._write_npy(arr, scaled_parameter_path)

        return arr

    @classmethod
    def _save_model(cls, format='mat'):  # 'npy', 'binary', 'pickle', 'segy'?
        pass

#   def load_data(self,fnames):
#
#       models = dict()
#
#       for model_name, fname in fnames.items():
#           m = spio.loadmat(fname)
#
#           # correct for some savemat stupidity
#           m['pixel_scale'].shape = -1,
#           m['pixel_unit'].shape =  -1,
#           m['domain_size'].shape =  -1,
#           m['domain_units'].shape =  -1,
#           m['scale_factor'].shape =  -1,
#           m['sz'].shape =  -1,
#
#           models[model_name] = m
#
#       return models
