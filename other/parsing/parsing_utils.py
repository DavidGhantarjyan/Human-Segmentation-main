import os
def _check_req(value, required):
    if required and value is None:
        raise LookupError("At least one of required arguments was not given")
    else:
        return value


def _typecheck(v, tp):
    if isinstance(tp, (list, tuple)):
        return any([isinstance(v, t) for t in tp])
    return isinstance(v, tp)


def is_type_of(value, tp=str, req=True):
    _check_req(value, req)
    if value is not None and not _typecheck(value, tp):
        raise TypeError(f"{value} ({type(value)}) is not a valid {tp}")
    else:
        return value


def is_range(value, fr=0, to=1, tp=(int, float), req=True):
    if tp is not None:
        is_type_of(value, tp, req)
    else:
        _check_req(value, req)
    if value is None:
        return value
    if fr < to:
        if not (fr <= value <= to):
            raise ValueError(f"{value} is out of range ({fr} to {to})")
    return value


def parse_range(lst, first_range, second_range, order=True, req=True):
    _check_req(lst, req)
    if lst is None:
        return None, None
    try:
        if len(lst) != 2:
            raise IndexError("Range must have exactly 2 numeric items")
        s, e = lst[0], lst[1]
    except Exception as err:
        raise TypeError(
            f"Range must be in the form [start, end]. The form you specified caused the following error: {str(err)}")
    is_range(s, *first_range)
    is_range(e, *second_range)
    if order and s > e:
        raise TypeError(
            f"In [start, end], start({s}) cannot be greater than end({e})")
    return s, e


def split_path(path):
    normalized_path = os.path.normpath(path)
    components = normalized_path.split(os.sep)
    return components

def construct_full_path(base_dir, *path_parts):
    full_path = os.path.join(base_dir, *path_parts)
    return os.path.normpath(full_path)