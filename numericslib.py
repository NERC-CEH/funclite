"""basic number related helper functions"""
import math as _math

def is_int(s: any) -> bool:
    """
    Test if value looks like an int

    Args:
        s (any): Value to test

    Returns:
        bool: True if looks like int, else False

    Examples:
        >>> is_int('1')
        True
        >>> is_int('2.33')
        False
        >>> is_int('A.1')
        False
    """
    try:
        n = int(s)
        f = float(s)
        return n == f
    except:
        return False


def translate_scale(val_in, out_min, out_max, val_in_max):
    """(float, float, float, float) -> float
    Translate val_in to a different scale range.

    val_in: the value to convert to new range
    out_min: the minimum value of the target range
    out_max: the max value of the target range
    val_in_max: the maximum value of the input range

    Example:
    Standardise a welsh city population value to lie between 0 and 1
    Bangor population = 5000, maximum population=100,000
    >>>translate_scale(5000, 0, 1, 100000)
    0.05
    """
    return val_in*(out_max - out_min)*(1/val_in_max) + out_min


def is_float(test: any) -> bool:
    """
    Return true of false if s is a float

    Args:
        test (any): Value to test

    Returns:
        bool: True of s can evaluate as a float

    Examples:
        >>> is_float('1')
        True
        >>> is_float('2.33')
        True
        >>> is_float('A.1')
        False
    """
    try:
        float(test)
        return True
    except ValueError:
        return False


def roundx(v):
    """(float)->float
    round to the more extreme value
    """
    if v < 0:
        return _math.floor(v)
    return _math.ceil(v)


def round_normal(x, y=0):
    """ A classical mathematical rounding by Voznica """
    m = int('1'+'0'*y) # multiplier - how many positions to the right
    q = x*m # shift to the right by multiplier
    c = int(q) # new number
    i = int( (q-c)*10 ) # indicator number on the right
    if i >= 5:
        c += 1
    return c/m



def hex2rgb(v):
    """(iter|str) -> list of tuples
    convert hex to decimal
    """
    if isinstance(v, str):
        v = [v]
    v = [s.lstrip('#') for s in v]
    out = []
    for h in v:
        out.append((tuple(int(h[i:i+2], 16) for i in (0, 2, 4))))
    return out


def is_numeric(v: any) -> bool:
    """
    Can v be evaluated as a numeric.

    Args:
        v ():

    Returns:
        bool: True if is_int or is_float

    Notes:
        Calls is_int and is_float

    Examples:
        >>> is_numeric('A')
        False
        >>> is_numeric('1')
        True
        >>> is_numeric('2.33')
        True
    """
    return is_int(v) and is_float(v)


