from pysit.util.basic_registration_factory import TernaryRegistrationFactory

__docformat__ = "restructuredtext en"

__all__ = ['SolverFactory',
           'SolverPhysicsMismatchError',
           'SolverDynamicsMismatchError']


class SolverFactory(TernaryRegistrationFactory):
    """ Factory for constant density acoustic wave (time-domain) solvers.

    Widgets (classes) can be registered with an instance of this class.
    Arguments to the factory's `__call__` method are then passed to a function
    specified by the registered factory, which validates the input and returns
    a instance of the class that best matches the inputs.

    Attributes
    ----------

    registry : dict
        Dictionary mapping classes (key) to function (value) which validates
        input.

    default_widget_type : type
        Class of the default widget.  Defaults to None.

    validation_functions : list of strings
        List of function names that are valid validation functions.

    Parameters
    ----------

    default_widget_type : type, optional

    additional_validation_functions : list of strings, optional
        List of strings corresponding to additional validation function names.

    Methods
    -------

    register
        Method for registering a class with the factory.

    Notes
    -----

    * A valid validation function must be a classmethod of the registered widget
      and it must return True or False.

    """

    supports = {'equation_physics': None,  # 'constant-density-acoustic', 'elastic'
                'equation_dynamics': None}  # 'time', 'frequency', 'laplace'

    def _check_registered_widget(self, *args, **kwargs):
        """ Implementation of a basic check to see if arguments match a widget."""

        # Update the defaults from the kwargs
        new_kwargs = self.supports.copy()
        new_kwargs.update(kwargs)

        # It might be wise, at some point, to pop
        return TernaryRegistrationFactory._check_registered_widget(self, *args, **new_kwargs)

    def register(self, WidgetType, validation_function=None, is_default=False):
        """ Register a widget with the factory.

        If `validation_function` is not specified, tests `WidgetType` for
        existence of any function in in the list `self.validation_functions`,
        which is a list of strings which must be callable class attribut

        Parameters
        ----------

        WidgetType : type
            Widget to register.

        validation_function : function, optional
            Function to validate against.  Defaults to None, which indicates
            that a classmethod in validation_functions is used.

        is_default : bool, optional
            Sets WidgetType to be the default widget.

        """

        if self.supports['equation_physics'] != WidgetType.supports['equation_physics']:
            err_str = ("Factory {0} accepts '{1}' physics while solver {2} "
                       "supports '{3}' physics.".format(self.__class__.__name__,
                                                        self.supports['equation_physics'],
                                                        WidgetType.__name__,
                                                        WidgetType.supports['equation_physics']))
            raise SolverPhysicsMismatchError(err_str)
        elif self.supports['equation_dynamics'] != WidgetType.supports['equation_dynamics']:
            err_str = ("Factory {0} accepts '{1}' dynamics while solver {2} "
                       "supports '{3}' dynamics.".format(self.__class__.__name__,
                                                        self.supports['equation_dynamics'],
                                                        WidgetType.__name__,
                                                        WidgetType.supports['equation_dynamics']))
            raise SolverDynamicsMismatchError(err_str)
        else:
            TernaryRegistrationFactory.register(self, WidgetType,
                                              validation_function=validation_function,
                                              is_default=is_default)


class SolverPhysicsMismatchError(ValueError):
    """ Exception for when a factory's equation physics doesn't match a
        solver's supported physics. """


class SolverDynamicsMismatchError(ValueError):
    """ Exception for when a factory's equation dynamics doesn't match a
        solver's supported dynamics. """
