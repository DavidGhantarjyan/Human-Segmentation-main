import os
def _check_req(value, required):
    """
    Check if a required value is provided.

    :param value: The value to check.
    :param required: Boolean flag indicating whether the value is required.
    :return: The value if it is provided.
    :raises LookupError: If the value is required but None is provided.
    """
    if required and value is None:
        raise LookupError("At least one of required arguments was not given")
    else:
        return value


def _typecheck(v, tp):
    """
    Check if a value is an instance of a given type or one of a list of types.

    :param v: The value to check.
    :param tp: A type or tuple/list of types to check against.
    :return: True if v is an instance of one of the types, False otherwise.
    """
    if isinstance(tp, (list, tuple)):
        return any([isinstance(v, t) for t in tp])
    return isinstance(v, tp)


def is_type_of(value, tp=str, req=True):
    """
    Validate that a given value is of the expected type.

    :param value: The value to validate.
    :param tp: Expected type or a list/tuple of types (default is str).
    :param req: Boolean flag indicating whether the value is required (default is True).
    :return: The original value if it passes the type check.
    :raises TypeError: If the value is not of the expected type.
    """
    _check_req(value, req)
    if value is not None and not _typecheck(value, tp):
        raise TypeError(f"{value} ({type(value)}) is not a valid {tp}")
    else:
        return value


def is_range(value, fr=0, to=1, tp=(int, float), req=True):
    """
    Validate that a numeric value falls within a specified range.

    :param value: The numeric value to validate.
    :param fr: The lower bound of the range (inclusive).
    :param to: The upper bound of the range (inclusive).
    :param tp: Expected numeric type (default is int or float).
    :param req: Boolean flag indicating whether the value is required (default is True).
    :return: The original value if it is within range.
    :raises TypeError: If the value is not of the expected type.
    :raises ValueError: If the value is outside the specified range.
    """
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
    """
    Parse and validate a range given as a list or tuple of two numeric values.

    :param lst: The list or tuple containing the range [start, end].
    :param first_range: The acceptable range for the start value.
    :param second_range: The acceptable range for the end value.
    :param order: Boolean flag indicating whether the first value should be less than or equal to the second (default True).
    :param req: Boolean flag indicating whether the range is required (default True).
    :return: A tuple (start, end) if valid.
    :raises IndexError: If the list does not have exactly two items.
    :raises TypeError: If the values do not conform to the expected numeric ranges or order.
    """
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

def parse_four_element_range(lst, x_range, y_range, order=True, req=True):
    """
    Parse and validate a list or tuple of four numeric values [x_min, y_min, x_max, y_max].

    :param lst: The list or tuple containing the values [x_min, y_min, x_max, y_max].
    :param x_range: A tuple (x_min_range, x_max_range) where each is a tuple (fr, to) defining the acceptable range for x_min and x_max.
    :param y_range: A tuple (y_min_range, y_max_range) where each is a tuple (fr, to) defining the acceptable range for y_min and y_max.
    :param order: Boolean flag indicating whether x_min <= x_max and y_min <= y_max must hold (default True).
    :param req: Boolean flag indicating whether the list is required (default True).
    :return: A tuple (x_min, y_min, x_max, y_max) if valid.
    :raises LookupError: If the list is required but None is provided.
    :raises IndexError: If the list does not have exactly four items.
    :raises TypeError: If the values do not conform to the expected numeric types or ranges.
    :raises ValueError: If the order condition (x_min <= x_max, y_min <= y_max) is not met when order=True.
    """
    # Проверяем, что lst предоставлен, если требуется
    _check_req(lst, req)
    if lst is None:
        return None, None, None, None

    if not isinstance(lst, (list, tuple)):
        raise TypeError(f"Expected a list or tuple, got {type(lst)}")
    if len(lst) != 4:
        raise IndexError("List must have exactly 4 numeric items: [x_min, y_min, x_max, y_max]")

    try:
        x_min, y_min, x_max, y_max = lst
    except Exception as err:
        raise TypeError(
            f"List must be in the form [x_min, y_min, x_max, y_max]. The form you specified caused the following error: {str(err)}")

    x_min_range, x_max_range = x_range
    y_min_range, y_max_range = y_range

    is_range(x_min, *x_min_range, tp=(int, float), req=True)
    is_range(y_min, *y_min_range, tp=(int, float), req=True)
    is_range(x_max, *x_max_range, tp=(int, float), req=True)
    is_range(y_max, *y_max_range, tp=(int, float), req=True)

    if order:
        if x_min > x_max:
            raise ValueError(f"In [x_min, y_min, x_max, y_max], x_min ({x_min}) cannot be greater than x_max ({x_max})")
        if y_min > y_max:
            raise ValueError(f"In [x_min, y_min, x_max, y_max], y_min ({y_min}) cannot be greater than y_max ({y_max})")

    return (x_min, y_min, x_max, y_max)

def split_path(path):
    """
    Split a file path into its components.

    :param path: The file path as a string.
    :return: A list of path components.
    """
    normalized_path = os.path.normpath(path)
    components = normalized_path.split(os.sep)
    return components

def construct_full_path(base_dir, *path_parts):
    """
    Construct a normalized full file path from a base directory and additional path components.

    :param base_dir: The base directory.
    :param path_parts: Additional components of the path.
    :return: A normalized full path.
    """
    full_path = os.path.join(base_dir, *path_parts)
    return os.path.normpath(full_path)

def parse_linspace(lst, first_range, second_range, count_range, order=True, req=True):
    _check_req(lst, req)
    if lst is None:
        return None, None
    try:
        if len(lst) != 3:
            raise IndexError("Range must have exactly 3 numeric items")
        s, e, c = lst[0], lst[1], lst[2]
    except Exception as err:
        raise TypeError(
            f"Range must be in the form [start, end, count]. The form you specified caused the following error: {str(err)}")
    is_range(s, *first_range)
    is_range(e, *second_range)
    is_range(c, *count_range)
    if order and s > e:
        raise TypeError(
            f"In [start, end, count], start({s}) cannot be greater than end({e})")
    return s, e, c


