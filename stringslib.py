# pylint: skip-file
"""string manipulations and related helper functions"""

# base imports
import re as _re
import numbers
import random as _random
import string as _string
import datetime as _datetime
import time as _time
from typing import Any as _Any
from typing import List as _List

# my imports
import funclite.numericslib
from funclite.numericslib import round_normal as _rndnorm

ascii_punctuation = ['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '¦', '}', '~', "'"]
ascii_punctuation_strict = ['!', '"', '(', ')', ',', '-', '.', ':', ';', '?', "'"]
ascii_and = ['&', '+']
ascii_or = ['|']


def plus_minus():
    """get plus minus"""
    return u"\u00B1"


def pretty_date_time_now() -> str:
    """Current datetime as pretty string
    Returns:
        (str): Pretty now datetime

    Examples:
        >>> pretty_date_time_now()
        '12:59 GMT on 12 Feb 2021'
    """
    return _time.strftime('%H:%M%p %Z on %b %d, %Y')


def non_breaking_space2space(s, replace_with=' '):
    """replace non breaking spaces"""
    return s.replace('\xc2\xa0', replace_with)


class Visible:
    visible_strict_with_space = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
    visible_strict_sans_space = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

    @staticmethod
    def ord_dict(with_space=False):
        """(bool) -> dict
        Get dictionary of printable chars
        with their ord number as the key
        """
        s = Visible.visible_strict_with_space if with_space else Visible.visible_strict_sans_space
        dic = {ord(value): value for value in s}
        return dic


def list_to_readable_str(seq: _List[_Any]) -> str:
    """
    Convert a list into a grammatically correct human readable string (with an Oxford comma).

    Args:
        seq: list

    Returns: str

    Examples:
        >>> list_to_readable_str('A', 2)
        'A and 2'

        >>> list_to_readable_str(['A', 'B,B', 'C,C,C'])
        'A, B,B, and C,C,C'
    """
    # Ref: https://stackoverflow.com/a/53981846/
    seq = [str(s) for s in seq]
    if len(seq) < 3:
        return ' and '.join(seq)
    return ', '.join(seq[:-1]) + ', and ' + seq[-1]


def iter_member_in_str(s: str, match: (str, tuple, list), ignore_case: bool = False) -> bool:
    """
    Check if any member of an iterable is IN s

    Args:
        s (str): string to check list items against
        match (list, str, tuple): items to check for being IN s
        ignore_case (bool): make check case insensitive

    Returns:
        bool: True if match in [ [],(),None,'',0 ] or if any item in member is IN s else False
    """
    if not match: return True  # everything is a match if we have nothing to match to
    if type(match) is str:
        return match in s
    if ignore_case: s = s.lower()
    for m in match:
        if ignore_case:
            m = m.lower()
        if m in s: return True
    return False


def str_in_iter(s: str, iter_: (list, tuple), ignore_case: bool = True) -> bool:
    """
    Check if s occurs in any member of iter

    Args:
        s (str): The string to check against iter
        iter_ (list, tuple): The iterable to test if s occurs in any member
        ignore_case (bool): Ignore case

    Returns:
        bool: True if occurs else False OR if iter_ is empty or None
    """
    def _filt(t: str):
        if ignore_case:
            return s.lower() in t.lower()
        else:
            return s in t
    if not iter_: return True
    lst = list(filter(_filt, iter_))
    return any(lst)



def datetime_stamp(datetimesep=''):
    """(str) -> str
    Returns clean date-time stamp for file names etc
    e.g 01 June 2016 11:23 would be 201606011123
    str is optional seperator between the date and time
    """
    fmtstr = '%Y%m%d' + datetimesep + '%H%m%S'
    return _time.strftime(fmtstr)


def read_number(test, default=0):
    """(any,number) -> number
    Return test if test is a number, or default if s is not a number
    """
    if isinstance(test, str):
        if funclite.numericslib.is_float(test):
            return float(test)
        else:
            return default
    elif isinstance(test, numbers.Number):
        return test
    else:  # not a string or not a number
        return default


def rndstr(length):
    """(int) -> str
    Return random alphanumeric string of length l

    l:
        string length

    Example:
        >>>rndstr(3)
        A12
        >>>rndstr(5)
        DeG12
    """
    return ''.join(_random.choice(_string.ascii_uppercase + _string.ascii_lowercase + _string.digits) for _ in range(length))


def filter_alphanumeric1(s, encoding='ascii', strict=False, allow_cr=True, allow_lf=True, exclude=(), include=(),
                         replace_ampersand='and', remove_single_quote=False, remove_double_quote=False,
                         exclude_numbers=False, strip=False, fix_nbs=True):
    """
    Pass a whole string/bytes, does the whole string!

    Args:
        s (str): string to filter
        encoding (str): specify encoding of s, implicitly converts characters between encodings
        strict (bool): only letters and numbers are returned, space is allowed
        allow_cr, allow_lf: include or exclude CR
        allow_lf: include or exclude LF
        exclude (tuple): force exclusion of these chars
        include (tuple): force exclusion of these chars
        replace_ampersand: replace "&" with the argument
        remove_single_quote: remove single quote from passed string
        remove_double_quote: remove double quote from passed string

    Returns:
        str: the filtered string

    Examples:
        >>> filter_alphanumeric('asd^!2', strict=True)
        asd2
    """
    if not s: return s

    if isinstance(s, bytes):
        s = s.decode(encoding, 'ignore')

    if fix_nbs:
        s = non_breaking_space2space(s, ' ')

    if exclude_numbers:
        lst = list(exclude)
        _ = [lst.extend([i]) for i in range(10)]

    if remove_single_quote:
        s = s.replace("'", "")
    if remove_double_quote:
        s = s.replace('"', '')
    if replace_ampersand:
        s = s.replace('&', replace_ampersand)

    build = []
    for c in s:
        keep = filter_alphanumeric(c, strict, allow_cr, allow_lf, exclude, include)
        if keep:
            build.append(c)
    out = ''.join(build)
    if strip:
        out = out.lstrip().rstrip()
    return out


def filter_numeric1(s, encoding='ascii', is_numeric=('.',)):
    """(str, str) -> str
    Filter out everything but digits from s

    Parameters:
        is_numeric: tuple of characters considered numberic
        s: str to process
        encoding: a vald encoding string, e.g. 'utf8' or 'ascii' if isinstance(s, bytes)
    """
    if isinstance(s, bytes):
        s = s.decode(encoding, 'ignore')
    out = [c for c in s if filter_numeric(c, is_numeric=is_numeric)]
    return ''.join(out)


def filter_numeric(char, is_numeric=('.',)):
    """(char(1)) -> bool
    As filter_alphanumeric, but just digits.
    Expects a length 1 string

    Example:
    >>>filter_numeric('1')
    True

    >>>filter_numeric('a')
    False
    """
    return 48 <= ord(char) <= 57 or char in is_numeric


def filter_punctuation(s, exclude=('!', '?', '.'), replace=' '):
    """(str, iterable, str) -> str
    Replace punctuation
    s: strint to process
    exclude: list of punctuation to retain
    replace: replace punctuation matches with replace
    """
    out = ''
    for a in s:
        out += a if a not in _string.punctuation or a in exclude else replace
    return out


# region files and paths related
def filter_alphanumeric(char, strict=False, allow_cr=True, allow_lf=True, exclude=(), include=()):
    """(char(1), bool, bool, bool, bool, tuple, tuple) -> bool
    Use as a helper function for custom string filters.

    Note: Accepts a single char. Use filter_alphanumeric1 for varchar

    for example in scrapy item processors

    strict : bool
        only letters and numbers are returned

    allow_cr, allow_lf : bool
        include or exclude cr lf

    exclude,include : tuple(str,..)
        force true or false for passed chars

    Example:
    l = lambda x: _filter_alphanumeric(x, strict=True)
    s = [c for c in 'abcef' if l(c)]
    """
    if not char: return char
    if char in exclude: return False
    if char in include: return True

    if allow_cr and ord(char) == 13: return True
    if allow_lf and ord(char) == 10: return True

    if not allow_cr and ord(char) == 13: return False
    if not allow_lf and ord(char) == 10: return False

    if not char: return char
    if strict:
        return 48 <= ord(char) <= 57 or 65 <= ord(char) <= 90 or 97 <= ord(char) <= 122 or ord(char) == 32  # 32 is space
    else:
        return 32 <= ord(char) <= 126


def add_right(s, char='/'):
    """(str, str) -> str
    Appends suffix to string if it doesnt exist
    """
    s = str(s)
    if not s.endswith(char):
        return s + char
    else:
        return s


def add_left(s, char):
    """(str, str) -> str
    Appends prefix to string if it doesnt exist
    """
    s = str(s)
    if not s.startswith(char):
        return char + s
    else:
        return s


def trim(s, trim_=' '):
    """(str,str) -> str
    remove leading and trailing chars

    trim('12asc12','12)
    >>>'asc'
    """
    assert isinstance(s, str)

    while s[0:len(trim_)] == trim_:
        s = s.lstrip(trim_)

    while s[len(s) - len(trim_):len(s)] == trim_:
        s = s.rstrip(trim_)

    return s


def rreplace(s, match, replacewith, cnt=1):
    """(str,str,str,int)->str"""
    return replacewith.join(s.rsplit(match, cnt))


def get_between(s, first, last, to_end_if_no_last=False):
    """(str, str, str) -> str
    Gets text between first and last, searching from the left

    s:
        String to search
    first:
        first substring
    last:
        last substring

    Returns empty string if first was not matched, or if last was not matched and to_end_if_no_last = False
    """

    end_ = None
    try:
        start = s.index(first) + len(first)  # if no start, return empty string
    except:
        return ''

    try:
        end_ = s.index(last, start)
        return s[start:end_]
    except ValueError:
        if to_end_if_no_last:
            if not end_: end_ = len(s)
            return s[start:end_]
        return ''


def get_between_r(s, first, last, to_end_if_no_last):
    """(str, str, str) -> str
    Gets text between first and last, searching from the right
    i.e. first is looked for first from the right
    s:
        String to search
    first:
        first substring
    last:
        last substring

    Example:
    >>> get_between_r('11_YY_23423_XX_12', 'XX', 'YY')
    '_23423_'
    """
    try:
        end_ = s.rindex(first)
    except:
        return ''

    try:
        start = s.rindex(last) + len(last)
        return s[start:end_]
    except ValueError:
        if to_end_if_no_last:
            return s[0:end_]
        return ''


def to_ascii(s):
    """(byte|str) -> str

    Takes a string or bytes representation of
    a string and returns an ascii encoded
    string.
    """
    if isinstance(s, bytes):
        return s.decode('ascii', 'ignore')

    return s.encode('ascii', 'ignore').decode('ascii')


def newline_del_multi(s):
    """replaces multiple newlines with single one"""
    s = s.replace('\r', '\n')
    s = _re.sub('\n+', '\n', s)
    return s


# endregion


# time stuff
def time_pretty(seconds):
    """(float) -> str
    Return a prettified time interval
    for printing
    """
    sign_string = '-' if seconds < 0 else ''
    seconds = abs(_rndnorm(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%s%dd %dh %dm %ds' % (sign_string, days, hours, minutes, seconds)
    if hours > 0:
        return '%s%dh %dm %ds' % (sign_string, hours, minutes, seconds)
    if minutes > 0:
        return '%s%dm %ds' % (sign_string, minutes, seconds)

    return '%s%ds' % (sign_string, seconds)


def date_str_to_iso(s, fmt='%d/%m/%Y %H:%M'):
    """Return ISO formatted date as string
    Set fmt according to string input, as described at
    http://strftime.org/

    The default set is uk format, eg 1/12/2019 12:23

    Args:
        s (str): a string representation of the date
        fmt (str): the date format of string "s"

    Returns:
        str: Iso formatted date string

    Examples:
        >>> date_str_to_iso('1/5/2019 12:13')
        '20190501 12:13:00'
    """
    return _datetime.datetime.strptime(s, fmt).strftime('%Y%m%d %H:%M:%S')


def pretty_date_now(sep='-', with_time=False, time_sep=':') -> str:
    """Date as a pretty ISO style string.

    Args:
        sep (str): Seperator to use
        with_time (bool): include time
        time_sep (str): time seperator

    Returns:
        str: The formatted date

    Examples:
        >>> pretty_date_now()
        2021-12-25
        >>> pretty_date_now(sep='')
        20211225
        >>> pretty_date_now(with_time=True)
        2021-12-25 12:39:44
    """
    if with_time:
        s = '%Y{}%m{}%d %H{}%M{}%S'.format(sep, sep, time_sep, time_sep)
    else:
        s = '%Y{}%m{}%d'.format(sep, sep)
    return _time.strftime(s)


def re_place(s, find_, with_):
    """fastest replace!"""
    return _re.sub(find_, with_, s)


def wordcnt(s):
    """count of words in s"""
    tokens = s.split()
    return len(tokens)


def index_all(s, substr, overlap=False):
    """return indexes of all occurences
    of substr in s
    """
    if overlap:
        return [m.start() for m in _re.finditer('(?=%s)' % substr, s)]

    return [m.start() for m in _re.finditer(substr, s)]


def numbers_in_str(s, type_=float):
    """(str) -> list

    Return list of numbers in s, PROVIDED numbers are positive integers
    s: the string
    type_: type to convert number to (e.g. float)
    
    Example:
    >>>numbers_in_str('asda 1 ssad', type_=float)
    [1.0]

    >>>numbers_in_str('asda 1.23 ssad', type_=int)
    []   #i.e. doesnt get floats
    """
    if not s: return []
    s = filter_numeric1(s, is_numeric=('.', ' '))
    return [type_(ss) for ss in s.split() if ss.isdigit()]


def numbers_in_str2(s, type_=float):
    """(str) -> list

    Return list of numbers in s, works with floats

    parameters:
        s: the string
        type_: type to convert number to (e.g. float)
    
    Example:
    >>>numbers_in_str('asda 1.23 ssad 2.1', type_=float)
    [1.23, 2.1]

    >>>numbers_in_str('asda ssad', type_=int)
    []
    """
    lst = []
    if not s: return []
    s = filter_numeric1(s, is_numeric=('.', ' '))
    for t in s.split():
        try:
            lst.append(type_(t))
        except ValueError:
            pass
    return lst


def join_newline(lst: list, joinstr: str = ',', every_n: int = 3, newline='\n'):
    """

    Args:
        lst (list): a list to concatenate
        joinstr (str): string to use for join
        every_n (int): use newline char every every_n in list
        newline (str): newline char, substitute as required, e.g. newline='\n\n' for double spacing

    Returns: (str): The joined string

    Examples:
         >>> join_newline(['a','b','c','d'], ',', 2)
         'a,b
         c,d'
    """
    if not every_n:
        return joinstr.join(lst)

    for i in range(len(lst)):
        if i > 0 and i % every_n == 0:
            lst[i] = '%s%s' % (newline, str(lst[i]))

    return joinstr.join(lst)



if __name__ == '__main__':
    # dirty testing
    b = str_in_iter('s', ['a', 'a'])
    assert b is False
    b = str_in_iter('s', ['s', 'a'])
    assert b is True