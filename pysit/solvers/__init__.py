from .wave_factory import *
from .helmholtz_factory import *
from .model_parameter import * #So we can set bounds like WaveSpeed.add_upper_bound(val)

#__all__ = filter(lambda s: not s.startswith('_'),dir())
