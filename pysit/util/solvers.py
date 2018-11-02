from collections import OrderedDict

__all__ = ['supports', 'inherit_dict']


def supports(needle, haystack):

    if isinstance(needle, str):
        if isinstance(haystack, str):
            return needle == haystack
        else:
            return needle in haystack

    elif isinstance(needle, int):
        if isinstance(haystack, int):
            return needle == haystack
        else:
            return needle in haystack
    else:
        raise ValueError('Needle does not match known requirement type.')


def inherit_dict(name, local_name=None):

    # If a local name isn't specifed, use _<name>.
    if local_name is None:
        local_name = '_' + name

    def wrap_cls(cls):

        # always overwrite existing name dict, everything local should come
        # from the local_name dict
        new_dict = OrderedDict()

        for base in cls.__bases__:
            base_dict = getattr(base, name, dict())
            new_dict.update(base_dict)

        _new_dict = getattr(cls, local_name, dict())
        new_dict.update(_new_dict)

        setattr(cls, name, new_dict)

        return cls
    return wrap_cls
