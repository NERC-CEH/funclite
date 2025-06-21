# pylint: skip-file
"""string manipulations and related helper functions"""

# base imports
from abc import ABC as _ABC
import os.path as _path
import re as _re
import numbers
import random as _random
import string  # dont underscore, we may wish to access it on importing stringslib
import base64 as _base64

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

class Characters(_ABC):
    """ Special utf-8 characters, commonly used in math and written language
    """
    class Language(_ABC):
        copyright = '©'
        emdash = '—'
        endash = '–'
        non_breaking_space = '\xc2\xa0'
        registered_trademark = '®'
        superscript_1 = '¹'
        superscript_2 = '²'
        superscript_3 = '³'
    
    class NonPrinting(_ABC):
        cr = '\x0C'
        lf = '\n'
        crlf = cr + lf
        tab = '\t'
    
    class Math(_ABC):
        degree = '°'
        division_sign = '÷'
        approx_double_bar = '≈'
        fraction_half = '½'
        fraction_three_quarters = '¾'
        fration_quarter = '¼'
        plus_minus = '±'
        product_dot = '·'


def encode_b64(s: str, on_if_not: str = '') -> str:
    """
    Encode a string to base64

    Args:
        s (str): string to encode
        on_if_not (any): value to return when "if not s" is True. i.e. Default if s looks empty

    Returns:
        str: base64 encoded string

    Examples:
        >>> encode_b64('sassadad')
        'c2Fzc2FkYWQ='
        \nNow using if_not
        >>> encode_b64(None, on_if_not='this_was_none')  # noqa
        'this_was_none'
    """
    if not s:
        return on_if_not
    return _base64.b64encode(bytes(s, 'utf-8')).decode('utf-8')


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
    """
    Single static method returning an ordered dictionary of printable characters
    The dictionary is ordered by ord(<char>).

    Notes:
        Superseeded by stringslib.VisibleDict and stringslib.VisibleStrict dictionary global variables.
        This is left here to not break any old code.

    Examples:
        >>> Visible.ord_dict()[49],Visible.ord_dict()[97],Visible.ord_dict()[35]
        '1', 'a', '#'
    """
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

# See class Visible, these are easier to use
VisibleDict = {ord(value): value for value in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '}
VisibleStrict = {ord(value): value for value in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'}


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
    Check if any member of the iterable match occurs in string s

    Args:
        s (str): string to check list items against
        match (list, str, tuple): items to check for being IN s
        ignore_case (bool): make check case insensitive

    Returns:
        bool: True if match in [ [],(),None,'',0 ].
        i.e. if we pass nothing to match, then call it a match.
        True if any item in match is IN s, else False.

    Examples:
        >>> iter_member_in_str('this_is_a_TEST', ['a_test'], ignore_case=False)
        False
        \nIgnoring case....
        >>> iter_member_in_str('this_is_a_TEST', ['a_test'], ignore_case=True)
        True
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


def rndstr(length=8, from_=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    """(int) -> str
    Return random alphanumeric string of length l

    Args:
        length (int): String length
        from_ (str): String to generate the random string from

    Examples:
        >>> import string
        >>> rndstr(3)
        'A12'
        >>> rndstr(5, from_=string.ascii_lowercase)
        'twgvy'
        >>> rndstr(5, from_='AB')
        'AABA'
    """
    return ''.join(_random.choice(from_) for _ in range(length))


get_random_string = rndstr  # noqa Alternative name, I can never find rndstr

def filter_alphanumeric1(s, encoding='ascii', strict=False, allow_cr=True, allow_lf=True, exclude=(), include=(),
                         replace_ampersand='and', remove_single_quote=False, remove_double_quote=False,
                         exclude_numbers=False, strip=False, fix_nbs=True):
    """
    Pass a whole string/bytes, does the whole string!

    Args:
        s (str): string to filter
        encoding (str): specify encoding of s, implicitly converts characters between encodings
        strict (bool): only letters and numbers are returned, space is allowed
        allow_cr: include or exclude CR
        allow_lf: include or exclude LF
        exclude (tuple): force exclusion of these chars
        include (tuple): force inclusion of these chars
        replace_ampersand: replace "&" with the argument
        remove_single_quote: remove single quote from passed string
        remove_double_quote: remove double quote from passed string
        exclude_numbers: exclude digits
        strip: strip spaces
        fix_nbs: Replace none breaking spaces with spaces, which can then be filtered according to other arguments

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

def duplicate_char_remove(s:str, char: str = ' ') -> str:
    """ clean duplicate chars from s

    Args:
        s: string to clean
        char: duplicate char to remove

    Returns:
        Cleaned string, or s if not s, e.g. None if s is None

    Examples:
        >>> duplicate_char_remove('asd!!  !!sadf', char='!')
        'asd^!2  !sadf'
    """
    if not s: return s
    while [char] * 2 in s:
        s = s.replace([char] * 2, char)
    return s

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
        out += a if a not in string.punctuation or a in exclude else replace
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
        force true or false for passed chars. Include means we KEEP the char.

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


def replace_dup(s: str, char: str, try_limit: int = 100) -> str:
    """
    Replace duplicate chars

    Args:
        s: string
        char: char to remove duplicates, does work with strings > 1
        try_limit: Limit of times to replace - a safety net

    Raises:
        RecursionError: If number of replacement attemps > try_limit

    Returns:
        The string with duplicates of string "char" removed.

    Examples:

        Single char

        >>> replace_dup('tthis is ttttttthick', 't')
        'this is thick'
    """
    n = 0
    while True:
        if not s: return ''
        i = len(s)
        s = s.replace(char*2, char)
        if i == len(s): break
        n += 1
        if n > 100:
            raise RecursionError('Recursion error when replacing duplicate string "%s" in input string "%s"' % (char, s))
    return s

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


def get_splits(s, grab: tuple, split_by: str = '_', out_sep: str = '_', exclude_extension: bool = False, filter_invalid_grab: bool = False):
    """
    Split a string, grab the results by index, then glue it back together with out_sep.

    Args:
        s: String to split
        grab: Tuple (or list) if the indexes to grab back
        split_by: The string to split the input string with
        out_sep: The seperator to use for the grabbed substrings
        exclude_extension: If True, the extension will be removed from the string before the split, otherwise the entire file name will be sued
        filter_invalid_grab: If the split string "s" does not have an index matched in grab, then ignore that grab index.

    Raises:
        IndexError: If filter_invalid_grab is False, and grab contains an index greater than len(grab) - 1

    Returns:
        Grabbed string elements seperated by out_sep.
        Returns "s" if "s" does not contain split_by
        Returns empty string if "not grab" evaluates to True

    Notes:
        Case sensitive

    Examples:

        Do not exclude extension (default)

        >>> get_splits('the.quick.brown.fox', (1, 2), '.', out_sep='_')
        'quick_brown'


        Exclude extension

        >>> get_splits('the_quick_brown.fox', (1, 2), '_', out_sep='_')
        'quick_brown.fox'


        Invalid grab indexes, filter_invalid_grab is False

        >>> get_splits('the_quick_brown.fox', (1, 10), '_', out_sep='_')
        IndexError: string index out of range


        Invalid grab indexes, filter_invalid_grab is True

        >>> get_splits('the_quick_brown.fox', (1, 2, 10), '_', out_sep='_')
        'quick'
    """
    def _get_ele(ss):
        if max(grab) > len(ss) - 1 and filter_invalid_grab:
            arr = list(filter(lambda x: x <= len(grab), grab))
        else:
            arr = [ss[i] for i in grab]
        return arr

    if split_by not in s:
        return s

    if not grab:
        return ''

    if exclude_extension:
        # We ignore the extension, so parse it out and glue it back on at the end
        ext = _path.splitext(s)
        splt = ext[0].split(split_by)
        a = _get_ele(splt)
        return '%s%s' % (out_sep.join(a), ext[-1])

    splt = s.split(split_by)
    return out_sep.join(_get_ele(splt))


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


def argparse_args_pretty_print(args: ("argparse.Namespace", str, None)) -> str:  # noqa
    """
    Get a pretty string of arguments from an argparse Namespace instance.
    This is used to extract arguments passed to scripts on the command line to pass
    Args:
        args ("argparse.Namespace", str, None): An argparse Namespace instance. This is the object returned by parse_args, which stores all command line arguments

    Returns:
        str: Pretty printed string of args. Or args itself, if args is a string instance or not args evaluates to True

    Examples:
        >>> argparse_args_pretty_print('a=1')  # noqa
        'a=1'
        >>> argparse_args_pretty_print(None)  # noqa
        None

        Now imagine args is an instance of argparse.ArgumentParser().parse_args()
        >>> argparse_args_pretty_print(args)  # noqa
        'folder:C:/myfolder overwrite:True'
    """
    if isinstance(args, str) or not args: return args
    return ' '.join(['{0}:{1}'.format(k, v) for k, v in sorted(vars(args).items())])


# region time
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


def pretty_date_now(sep: str = '-', with_time: bool = False, time_sep: str = ':', date_time_sep: str = ' ') -> str:
    """Date as a pretty ISO style string.

    Args:
        sep (str): Seperator to use
        with_time (bool): include time
        time_sep (str): time seperator
        date_time_sep (str): seperator between data and time (only relevant when with_time = True

    Returns:
        str: The formatted date

    Examples:
        >>> pretty_date_now()
        2021-12-25
        \nNo month seperator
        >>> pretty_date_now(sep='')
        20211225
        \nInclude the time
        >>> pretty_date_now(with_time=True)
        2021-12-25 12:39:44
        \nSpecify date_time_sep
        >>> pretty_date_now(with_time=True, date_time_sep='*')
        2021-12-25*12:39:44
    """
    if with_time:
        s = '%Y{}%m{}%d{}%H{}%M{}%S'.format(sep, sep, date_time_sep, time_sep, time_sep)
    else:
        s = '%Y{}%m{}%d'.format(sep, sep)
    return _time.strftime(s)

def pretty_date(dt: _datetime.datetime, sep: str = '-', with_time: bool = False, time_sep: str = ':', date_time_sep: str = ' ') -> str:
    """Date as a pretty ISO style string.

    Args:
        dt (datetime): A datetime instance
        sep (str): Seperator to use
        with_time (bool): include time
        time_sep (str): time seperator
        date_time_sep (str): seperator between data and time (only relevant when with_time = True

    Returns:
        str: The formatted date

    Examples:
        >>> pretty_date_now()
        2021-12-25
        \nNo month seperator
        >>> pretty_date_now(sep='')
        20211225
        \nInclude the time
        >>> pretty_date_now(with_time=True)
        2021-12-25 12:39:44
        \nSpecify date_time_sep
        >>> pretty_date_now(with_time=True, date_time_sep='*')
        2021-12-25*12:39:44
    """
    if with_time:
        s = '%Y{}%m{}%d{}%H{}%M{}%S'.format(sep, sep, date_time_sep, time_sep, time_sep)
    else:
        s = '%Y{}%m{}%d'.format(sep, sep)
    return dt.strftime(s)

# endregion time
#
# -------------
#
# region re
def re_place(s, find_, with_):
    """fastest replace!"""
    return _re.sub(find_, with_, s)
# endregion re
#
# -------------
#
def wordcnt(s: str) -> int:
    """
    Simple count of words in s

    Args:
        s (str): The string

    Returns:
        int: The count of words
    """
    return len(s.split())

def index_all(s: str, substr: str, overlap: bool = False) -> list[int]:
    """return indexes of all occurences of substr in s

    Args:
        s (str): The string in which to find substrings
        substr (str): The substring to find indexes of in s
        overlap (bool): Allow overlaps in the search, think of substring 'aaa' and the string 'aaaaaaaaa'.

    Returns:
        list[int]: List of indexes of occurences of substr in str
    """
    if overlap:
        return [m.start() for m in _re.finditer('(?=%s)' % substr, s)]
    return [m.start() for m in _re.finditer(substr, s)]


def numbers_in_str(s: str, type_=int, allow_scientific_notation: bool = False, encoding='ascii') -> list[(float, int)]:
    """
    Return list of numbers in s. Should support integer, floats and scientific notation

    Credit:
        Adapted from https://stackoverflow.com/a/29581287/5585800

    Args:
        s (str): the string
        type_: type to convert number to (e.g. float). Can be any function, provided it returns a number.
        encoding: Decode s using this codepage if s is bytes instance

    Returns:
        list[(float, int)]: list of numbers in s, force to type "type_"
        empty list: if no numerics in s

    Examples:

        Force to float

        >>> numbers_in_str('asda 1.23 XX 9 ssad', type_=float)
        [1.23, 9.0]


        No numbers

        >>> numbers_in_str('asda ssad', type_=int)
        []


        Scientific notation and type_ is float

        >>> numbers_in_str('quick**1e4** brown', type_=float, allow_scientific_notation=True)
        [1e4]


        Scientific notation and type_ is int

        >>> numbers_in_str('quick**1e4** brown', allow_scientific_notation=True)
        [10000]
    """
    if not s: return []
    if isinstance(s, bytes):
        s = s.decode(encoding, 'ignore')

    rr = _re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)

    if rr:
        if allow_scientific_notation:
            if type_ is int: type_ = lambda i: int(float(i))  # little work around to support scientific notation and int
            return [type_(ss.strip('.')) for ss in rr]
        return [type_(ss.strip('.')) for ss in rr if 'e' not in ss.lower()]
    return []



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
    pass
    # dirty testing
    # b = str_in_iter('s', ['a', 'a'])
    # assert b is False
    # b = str_in_iter('s', ['s', 'a'])
    # assert b is True
    # _ = numbers_in_str('ATT1329_932XX2_2019_IMG_4437.jpg', int)
    # _ = numbers_in_str('ATT1329_932XX2_2019_IMG_4437.jpg', int)

    cs_files=['ATT1000_744XX3_photo4-20190821-133210.jpg','ATT1001_744XX2_photo1-20190820-140919.jpg','ATT1002_744XX2_photo2-20190820-140944.jpg','ATT1003_744XX2_photo3-20190820-141011.jpg','ATT1004_744XX2_photo4-20190820-141054.jpg','ATT1005_721XX2_photo1-20190820-091957.jpg','ATT1006_721XX2_photo2-20190820-092104.jpg','ATT1007_721XX2_photo3-20190820-092051.jpg','ATT1008_721XX2_photo4-20190820-092127.jpg','ATT1009_721XX4_photo1-20190819-113742.jpg','ATT100_55XX2_photo1-20190523-120831.jpg','ATT1010_721XX4_photo2-20190819-113805.jpg','ATT1011_721XX4_photo3-20190819-113837.jpg','ATT1012_721XX4_photo4-20190819-113924.jpg','ATT1013_721XX1_photo1-20190819-090155.jpg','ATT1014_721XX1_photo2-20190819-090210.jpg','ATT1015_721XX1_photo3-20190819-090228.jpg','ATT1016_721XX1_photo4-20190819-090247.jpg','ATT1017_721XX5_photo1-20190819-100717.jpg','ATT1018_721XX5_photo2-20190819-100737.jpg','ATT1019_721XX5_photo3-20190819-100755.jpg','ATT101_55XX2_photo2-20190523-120942.jpg','ATT1020_793XX5_photo1-20190827-101054.jpg','ATT1021_793XX5_photo2-20190827-101121.jpg','ATT1022_793XX4_photo1-20190827-103811.jpg','ATT1023_793XX4_photo2-20190827-103835.jpg','ATT1024_793XX3_photo1-20190827-112958.jpg','ATT1025_793XX3_photo2-20190827-113024.jpg','ATT1026_793XX2_photo1-20190827-133259.jpg','ATT1027_793XX1_photo1-20190827-125545.jpg','ATT1028_793XX1_photo2-20190827-125610.jpg','ATT1029_776XX6_photo1-20190828-123319.jpg','ATT102_55XX3_photo1-20190522-143616.jpg','ATT1030_776XX6_photo2-20190828-123353.jpg','ATT1031_776XX2_photo1-20190828-115604.jpg','ATT1032_776XX2_photo2-20190828-115624.jpg','ATT1033_776XX4_photo1-20190828-130408.jpg','ATT1034_776XX3_photo1-20190828-102700.jpg','ATT1035_776XX3_photo2-20190828-102727.jpg','ATT1036_776XX3_photo3-20190828-102746.jpg','ATT1037_776XX1_photo1-20190828-111555.jpg','ATT1038_776XX1_photo2-20190828-111615.jpg','ATT1039_735XX5_photo1-20190827-085443.jpg','ATT103_55XX3_photo2-20190522-143642.jpg','ATT1040_735XX5_photo2-20190827-085502.jpg','ATT1041_735XX5_photo3-20190827-085515.jpg','ATT1042_735XX4_photo1-20190827-094112.jpg','ATT1043_735XX4_photo2-20190827-094231.jpg','ATT1044_735XX4_photo3-20190827-094349.jpg','ATT1045_735XX4_photo4-20190827-094404.jpg','ATT1046_735XX3_photo1-20190827-103107.jpg','ATT1047_735XX3_photo2-20190827-103229.jpg','ATT1048_735XX2_photo1-20190827-112633.jpg','ATT1049_735XX2_photo2-20190827-112716.jpg','ATT104_55XX1_photo1-20190522-093311.jpg','ATT1050_735XX2_photo3-20190827-112914.jpg','ATT1051_735XX2_photo4-20190827-113001.jpg','ATT1052_735XX1_photo1-20190827-130134.jpg','ATT1053_735XX1_photo2-20190827-130232.jpg','ATT1054_735XX1_photo3-20190827-130305.jpg','ATT1055_718XX5_photo1-20190826-084312.jpg','ATT1056_718XX5_photo2-20190826-084349.jpg','ATT1057_718XX5_photo3-20190826-084426.jpg','ATT1058_718XX5_photo4-20190826-084451.jpg','ATT1059_718XX4_photo1-20190826-114204.jpg','ATT105_55XX1_photo2-20190522-093349.jpg','ATT1060_718XX4_photo2-20190826-114221.jpg','ATT1061_718XX4_photo3-20190826-114244.jpg','ATT1062_718XX3_photo1-20190826-110606.jpg','ATT1063_718XX3_photo2-20190826-110705.jpg','ATT1064_718XX3_photo3-20190826-110740.jpg','ATT1065_718XX3_photo4-20190826-110811.jpg','ATT1066_718XX2_photo1-20190826-092346.jpg','ATT1067_718XX2_photo2-20190826-092411.jpg','ATT1068_718XX1_photo1-20190826-100754.jpg','ATT1069_718XX1_photo2-20190826-100816.jpg','ATT106_55XX5_photo1-20190522-133023.jpg','ATT1070_718XX1_photo3-20190826-100857.jpg','ATT1071_701X5_photo1-20190823-093811.jpg','ATT1072_701X5_photo2-20190823-093900.jpg','ATT1073_701X5_photo3-20190823-093936.jpg','ATT1074_701X5_photo4-20190823-094007.jpg','ATT1075_701X4_photo1-20190823-133552.jpg','ATT1076_701X4_photo2-20190823-133511.jpg','ATT1077_701X4_photo3-20190823-133520.jpg','ATT1078_701X4_photo4-20190823-133535.jpg','ATT107_57X1_photo1-2019.jpg','ATT1081_701X3_photo3-20190823-105144.jpg','ATT1082_701X2_photo1-20190823-112237.jpg','ATT1083_701X2_photo2-20190823-112259.jpg','ATT1084_701X2_photo3-20190823-112330.jpg','ATT1085_701X2_photo4-20190823-112447.jpg','ATT1086_701X1_photo1-20190823-123334.jpg','ATT1087_701X1_photo2-20190823-123359.jpg','ATT1088_701X1_photo3-20190823-123435.jpg','ATT1089_701X1_photo4-20190823-123508.jpg','ATT108_57X1_photo2-2019.jpg','ATT1090_702X5_photo1-20190822-085437.jpg','ATT1091_702X5_photo2-20190822-085528.jpg','ATT1092_702X4_photo1-20190822-122325.jpg','ATT1093_702X4_photo2-20190822-122409.jpg','ATT1094_702X4_photo3-20190822-122446.jpg','ATT1095_702X3_photo1-20190822-131345.jpg','ATT1096_702X3_photo2-20190822-131417.jpg','ATT1097_702X3_photo3-20190822-131524.jpg','ATT1098_702X2_photo1-20190822-094644.jpg','ATT1099_702X2_photo2-20190822-094728.jpg','ATT109_57X1_photo3-2019.jpg','ATT1100_702X2_photo3-20190822-094759.jpg','ATT1101_702X1_photo1-20190822-113253.jpg','ATT1102_702X1_photo2-20190822-113317.jpg','ATT1103_702X1_photo3-20190822-113336.jpg','ATT1104_534XX5_photo1-20190828-091054.jpg','ATT1105_534XX5_photo2-20190828-091302.jpg','ATT1106_535XX9_photo1-20190829-093843.jpg','ATT1107_535XX9_photo2-20190829-093905.jpg','ATT1108_535XX6_photo1-20190828-142607.jpg','ATT1109_535XX6_photo2-20190828-142703.jpg','ATT110_57X2_photo1-20190521-122317.jpg','ATT1110_535XX6_photo3-20190828-142720.jpg','ATT1111_535XX8_photo1-20190828-141404.jpg','ATT1112_535XX8_photo2-20190828-141421.jpg','ATT1113_535XX7_photo1-20190828-134041.jpg','ATT1114_535XX7_photo2-20190828-134101.jpg','ATT1115_534XX3_photo1-20190827-151747.jpg','ATT1116_534XX3_photo2-20190827-151913.jpg','ATT1117_534XX2_photo1-20190827-135936.jpg','ATT1118_534XX2_photo2-20190827-135958.jpg','ATT1119_534XX2_photo3-20190827-140134.jpg','ATT111_57X2_photo2-20190521-122344.jpg','ATT1120_534XX1_photo1-20190827-130415.jpg','ATT1121_534XX1_photo2-20190827-130445.jpg','ATT1122_534XX1_photo3-20190827-130508.jpg','ATT1123_592X5_photo1-20190904-092503.jpg','ATT1124_592X5_photo2-20190904-093441.jpg','ATT1125_592X5_photo3-20190904-093545.jpg','ATT1126_592X2_photo1_2019.jpg','ATT1127_592X2_photo2_2019.jpg','ATT1128_592X4_photo1-20190904-102218.jpg','ATT1129_592X4_photo2-20190904-102317.jpg','ATT112_57X2_photo3-20190521-122426.jpg','ATT1130_592X3_photo1-20190904-110614.jpg','ATT1131_592X3_photo2-20190904-110730.jpg','ATT1132_592X1_photo1_2019.jpg','ATT1133_592X1_photo2_2019.jpg','ATT1134_542XX5_photo1_2019.jpg','ATT1135_542XX5_photo2_2019.jpg','ATT1136_542XX3_photo1_2019.jpg','ATT1137_542XX3_photo2_2019.jpg','ATT1138_542XX2_photo1_2019.jpg','ATT1139_542XX2_photo2_2019.jpg','ATT113_57X2_photo4-20190521-122505.jpg','ATT1140_542XX1_photo1_2019.jpg','ATT1141_542XX1_photo2_2019.jpg','ATT1142_542XX4_photo1_2019.jpg','ATT1143_542XX4_photo2_2019.jpg','ATT1144_542XX4_photo3_2019.jpg','ATT1145_542XX4_photo4_2019.jpg','ATT1146_791X5_photo1-20190903-120029.jpg','ATT1147_791X5_photo2-20190903-120057.jpg','ATT1148_791X4_photo1-20190904-092300.jpg','ATT1149_791X4_photo2-20190904-092321.jpg','ATT114_57X4_photo1-20190520-150445.jpg','ATT1150_791X3_photo1-20190904-102611.jpg','ATT1151_791X3_photo2-20190904-102649.jpg','ATT1152_791X2_photo1-20190903-093250.jpg','ATT1153_791X2_photo2-20190903-093320.jpg','ATT1154_791XX1_photo1-20190903-105520.jpg','ATT1155_692XX2_photo1-20190911-102324.jpg','ATT1156_692XX2_photo2-20190911-102128.jpg','ATT1157_692XX2_photo3-20190911-102157.jpg','ATT1158_692XX2_photo4-20190911-102246.jpg','ATT1159_692XX4_photo1-20190911-141555.jpg','ATT115_57X4_photo2-20190520-150545.jpg','ATT1160_692XX4_photo2-20190911-141618.jpg','ATT1161_692XX4_photo4-20190911-141745.jpg','ATT1162_692XX5_photo1-20190911-122600.jpg','ATT1163_692XX5_photo2-20190911-122621.jpg','ATT1164_692XX5_photo3-20190911-122646.jpg','ATT1165_692XX5_photo4-20190911-122710.jpg','ATT1166_1214XX4_photo1-20190910-101316.jpg','ATT1167_1214XX4_photo2-20190910-101341.jpg','ATT1168_1214XX4_photo3-20190910-101503.jpg','ATT1169_1214XX4_photo4-20190910-101520.jpg','ATT1170_1214XX5_photo1-20190910-113755.jpg','ATT1171_1214XX5_photo2-20190910-113849.jpg','ATT1172_1214XX5_photo3-20190910-113934.jpg','ATT1173_1214XX5_photo4-20190910-114006.jpg','ATT1174_1214XX2_photo1-20190909-120419.jpg','ATT1175_1214XX2_photo2-20190909-120441.jpg','ATT1176_1214XX2_photo3-20190909-120500.jpg','ATT1177_1214XX2_photo4-20190909-120517.jpg','ATT1178_1214XX3_photo1-20190909-101942.jpg','ATT1179_1214XX3_photo2-20190909-102002.jpg','ATT117_57X5_photo1-20190520-161312.jpg','ATT1180_1214XX3_photo3-20190909-102026.jpg','ATT1181_1214XX3_photo4-20190909-102058.jpg','ATT1182_1214XX1_photo1-20190909-091827.jpg','ATT1183_1214XX1_photo2-20190909-091843.jpg','ATT1184_1214XX1_photo3-20190909-091859.jpg','ATT1185_1214XX1_photo4-20190909-091920.jpg','ATT1187_569X3_photo1-20190910-103144.jpg','ATT1188_569X3_photo2-20190910-103454.jpg','ATT1189_569X3_photo3-20190910-103520.jpg','ATT118_57X5_photo2-20190520-161618.jpg','ATT1191_569X2_photo1-20190910-084429.jpg','ATT1192_569X2_photo2-20190910-084505.jpg','ATT1193_546XX7_photo1-20190911-111553.jpg','ATT1194_546XX7_photo2-20190911-111656.jpg','ATT1195_546XX6_photo1_2019.jpg','ATT1196_546XX6_photo2_2019.jpg','ATT1197_569X5_photo1-20190910-120650.jpg','ATT1198_569X5_photo2-20190910-120728.jpg','ATT1199_569X1_photo1-20190910-093829.jpg','ATT1200_569X1_photo2-20190910-093856.jpg','ATT1201_546XX1_photo1-20190911-095508.jpg','ATT1202_546XX1_photo2-20190911-095534.jpg','ATT1203_569X4_photo1-20190910-130616.jpg','ATT1204_569X4_photo2-20190910-130642.jpg','ATT1205_569X4_photo3-20190910-130708.jpg','ATT1206_501X5_photo1_2019.jpg','ATT1207_501X5_photo2_2019.jpg','ATT1208_501X4_photo1_2019.jpg','ATT1209_501X4_photo2_2019.jpg','ATT120_57X6_photo1_2019.jpg','ATT1210_501X4_photo3_2019.jpg','ATT1211_501X3_photo1_2019.jpg','ATT1212_501X3_photo2_2019.jpg','ATT1213_501X2_photo1-20190917-083807.jpg','ATT1214_501X2_photo2-20190917-083956.jpg','ATT1215_501X1_photo1_2019.jpg','ATT1216_501X1_photo2_2019.jpg','ATT1217_414X5_photo1-20190918-090514.jpg','ATT1218_414X5_photo2-20190918-090550.jpg','ATT1219_414X7_photo1-20190918-100540.jpg','ATT121_57X6_photo2_2019.jpg','ATT1220_414X7_photo2_2019.jpg','ATT1221_414X7_photo3_2019.jpg','ATT1222_414X1_photo1-20190917-145913.jpg','ATT1223_414X1_photo2-20190917-145940.jpg','ATT1224_414X1_photo3-20190917-150013.jpg','ATT1225_414X2_photo1-20190917-133452.jpg','ATT1226_414X2_photo2-20190917-133528.jpg','ATT1227_414X6_photo1_2019.jpg','ATT1228_414X6_photo2_2019.jpg','ATT1229_414X6_photo3_2019.jpg','ATT122_55XX4_photo1-20190522-161014.jpg','ATT1230_350XX1_photo1-20190909-164401.jpg','ATT1231_350XX1_photo2-20190909-164427.jpg','ATT1232_350XX1_photo3-20190909-164442.jpg','ATT1233_350XX1_photo4-20190909-164501.jpg','ATT1234_294X1_photo1-20190911-114819.jpg','ATT1235_294X1_photo2-20190911-114843.jpg','ATT1236_350XX4_photo1-20190909-115530.jpg','ATT1237_350XX4_photo2-20190909-115611.jpg','ATT1238_350XX4_photo3-20190909-115653.jpg','ATT1239_350XX2_photo1-20190909-145325.jpg','ATT123_55XX4_photo2-20190522-161209.jpg','ATT1240_350XX5_photo1-20190909-132318.jpg','ATT1241_350XX5_photo2-20190909-132502.jpg','ATT1242_350XX5_photo3-20190909-132611.jpg','ATT1243_350XX3_photo1-20190909-105143.jpg','ATT1244_350XX3_photo2-20190909-105231.jpg','ATT1245_350XX3_photo3-20190909-105302.jpg','ATT1246_294X5_photo1-20190910-130117.jpg','ATT1247_294X5_photo2-20190910-130156.jpg','ATT1248_294X5_photo3-20190910-130228.jpg','ATT1249_294X5_photo4-20190910-130304.jpg','ATT1250_294X4_photo1-20190910-105513.jpg','ATT1251_294X4_photo2-20190910-111613.jpg','ATT1252_294X4_photo3-20190910-111736.jpg','ATT1253_294X3_photo1-20190910-150154.jpg','ATT1254_294X3_photo2-20190910-150220.jpg','ATT1255_294X3_photo3-20190910-150321.jpg','ATT1256_294X2_photo1-20190911-094507.jpg','ATT1257_294X2_photo2-20190911-094539.jpg','ATT1258_294X2_photo3-20190911-094627.jpg','ATT1259_502X4_photo1-20190903-125733.jpg','ATT125_1121X1_photo1-20190527-144035.jpg','ATT1260_502X4_photo2-20190903-130122.jpg','ATT1261_502X1_photo1-20190902-174750.jpg','ATT1262_502X1_photo2-20190902-174833.jpg','ATT1263_502X2_photo1-20190903-102529.jpg','ATT1264_502X2_photo2-20190903-102604.jpg','ATT1265_502X2_photo3-20190903-102813.jpg','ATT1266_502X5_photo1-20190903-165054.jpg','ATT1267_502X5_photo2-20190903-165118.jpg','ATT1268_138XX1_photo1-20190916-135531.jpg','ATT1269_138XX1_photo2-20190916-135603.jpg','ATT126_1121X1_photo2-20190527-144147.jpg','ATT1270_116X2_photo1-20190917-135328.jpg','ATT1271_116X2_photo2-20190917-135350.jpg','ATT1272_116X3_photo1-20190917-122829.jpg','ATT1273_555X3_photo1-20190910-124206.jpg','ATT1274_555X3_sp_photo1-20190910-122150.jpg','ATT1275_555X4_photo1-20190909-144744.jpg','ATT1276_555X4_photo2-20190909-144822.jpg','ATT1277_555X5_photo1-20190909-125355.jpg','ATT1278_555X5_photo2-20190909-125415.jpg','ATT1279_555X2_photo1-20190910-092655.jpg','ATT127_1121X1_photo3-20190527-144229.jpg','ATT1280_555X2_photo2-20190910-092715.jpg','ATT1281_555X1_photo1-20190910-104320.jpg','ATT1282_555X1_photo2-20190910-104338.jpg','ATT1283_552XX4_photo1-20190904-133933.jpg','ATT1284_116X1_photo1-20190917-105837.jpg','ATT1285_138XX2_photo1-20190916-152322.jpg','ATT1286_138XX2_photo2-20190916-152400.jpg','ATT1287_576XX1_photo1-20190911-134728.jpg','ATT1288_576XX1_photo2-20190911-134916.jpg','ATT1289_576XX5_photo1-20190911-095834.jpg','ATT128_1130X1_photo1-20190522-111542.jpg','ATT1290_576XX5_photo2-20190911-095941.jpg','ATT1291_576XX3_photo1-20190911-125652.jpg','ATT1292_576XX3_photo2-20190911-125714.jpg','ATT1293_576XX2_photo1-20190911-105405.jpg','ATT1294_576XX2_photo2-20190911-105551.jpg','ATT1296_576XX4_photo1-20190911-151132.jpg','ATT1297_576XX4_photo2-20190911-151323.jpg','ATT1298_138XX5_photo1-20190916-124101.jpg','ATT1299_138XX5_photo2-20190916-124334.jpg','ATT129_1130X1_photo2-20190522-112110.jpg','ATT1300_138XX4_photo1-20190916-112308.jpg','ATT1301_138XX4_photo2-20190916-112451.jpg','ATT1302_138XX4_photo3-20190916-112558.jpg','ATT1303_138XX3_photo1-20190916-101457.jpg','ATT1304_138XX3_photo2-20190916-101528.jpg','ATT1305_552XX5_photo1-20190904-124559.jpg','ATT1306_552XX1_photo1-20190904-105452.jpg','ATT1307_552XX2_photo1-20190905-111146.jpg','ATT1308_552XX2_photo2-20190905-111618.jpg','ATT1309_552XX2_photo3-20190905-111809.jpg','ATT130_1130X1_photo3-20190522-111633.jpg','ATT1310_552XX3_photo1-20190904-144656.jpg','ATT1311_835X2_2019_IMG_4709.jpg','ATT1312_835X4_2019_IMG_4711.jpg','ATT1313_835X4_2019_IMG_4712.jpg','ATT1314_835X4_2019_IMG_4713.jpg','ATT1315_931X2_2019_IMG_3929.jpg','ATT1316_931X2_2019_IMG_3931.jpg','ATT1317_931X2_2019_IMG_3932.jpg','ATT1318_931X2_2019_IMG_3933.jpg','ATT1319_931X1_2019_IMG_3891.jpg','ATT1320_931X1_2019_IMG_3892.jpg','ATT1321_931X1_2019_IMG_3894.jpg','ATT1322_932XX1_2019_IMG_4372.jpg','ATT1323_932XX1_2019_IMG_4375.jpg','ATT1324_932XX1_2019_IMG_4377.jpg','ATT1325_932XX1_2019_IMG_4379.jpg','ATT1326_932XX1_2019_IMG_4381.jpg','ATT1327_932XX2_2019_IMG_4431.jpg','ATT1328_932XX2_2019_IMG_4433.jpg','ATT1329_932XX2_2019_IMG_4437.jpg','ATT132_1121X5_photo1-20190525-113304.jpg','ATT1330_932XX2_2019_IMG_4439.jpg','ATT1331_932XX3_2019_IMG_4444.jpg','ATT1332_932XX3_2019_IMG_4446.jpg','ATT1333_932XX3_2019_IMG_4448.jpg','ATT1334_932XX3_2019_IMG_4450.jpg','ATT1335_932XX3_2019_IMG_4452.jpg','ATT1336_932XX5_2019_IMG_4393.jpg','ATT1337_932XX5_2019_IMG_4394.jpg','ATT1338_932XX5_2019_IMG_4395.jpg','ATT1339_932XX5_2019_IMG_4396.jpg','ATT133_1121X5_photo2-20190525-113430.jpg','ATT1340_932XX5_2019_IMG_4397.jpg','ATT1341_932XX5_2019_IMG_4399.jpg','ATT1342_932XX4_2019_IMG_4425.jpg','ATT1343_932XX4_2019_IMG_4426.jpg','ATT1344_932XX4_2019_IMG_4427.jpg','ATT1345_932XX4_2019_IMG_4428.jpg','ATT1346_932XX4_2019_IMG_4429.jpg','ATT1347_983X4_2019_IMG_3105.jpg','ATT1348_983X4_2019_IMG_3107.jpg','ATT1349_1020X4_2019_IMG_2140.jpg','ATT1350_1020X4_2019_IMG_2142.jpg','ATT1351_1020X4_2019_IMG_2143.jpg','ATT1352_1041X5_2019_IMG_2258.jpg','ATT1353_1041X5_2019_IMG_2260.jpg','ATT1354_1041X3_2019_IMG_2240.jpg','ATT1355_1041X3_2019_IMG_2241.jpg','ATT1356_1041X3_2019_IMG_2243.jpg','ATT135_1121X2_photo1-20190527-114726.jpg','ATT136_1130X5_photo1-20190523-114540.jpg','ATT137_1130X5_photo2-20190523-114603.jpg','ATT138_1130X5_photo3-20190523-114944.jpg','ATT139_1130X5_photo4-20190523-115002.jpg','ATT140_1130X4_photo1-20190524-151607.jpg','ATT141_1130X4_photo2-20190524-151620.jpg','ATT142_1130X4_photo3-20190524-151657.jpg','ATT143_1130X4_photo4-20190524-151745.jpg','ATT145_1130X3_photo1-20190524-112423.jpg','ATT146_1130X3_photo2-20190524-112524.jpg','ATT147_1130X3_photo3-20190524-112638.jpg','ATT148_1130X2_photo2-20190522-144821.jpg','ATT149_1130X2_photo3-20190522-144951.jpg','ATT150_1130X2_photo4-20190522-145009.jpg','ATT151_825X2_photo1-20190528-092501.jpg','ATT152_825X2_photo2-20190528-092549.jpg','ATT153_825X1_photo1-20190528-103843.jpg','ATT154_825X1_photo2-20190528-104006.jpg','ATT156_825X4_photo1-20190528-115906.jpg','ATT157_825X4_photo2-20190528-120011.jpg','ATT158_825X5_photo1-20190528-131542.jpg','ATT159_825X5_photo2-20190528-131708.jpg','ATT160_825X3_photo1-20190522-094624.jpg','ATT161_825X3_photo2-20190522-094810.jpg','ATT163_17XX1_photo1-20190530-085845.jpg','ATT164_17XX1_photo2-20190530-090112.jpg','ATT165_17XX3_photo1-20190530-114822.jpg','ATT166_17XX3_photo2-20190530-114925.jpg','ATT167_19XX4_photo1-20190528-124822.jpg','ATT168_19XX4_photo2-20190528-124905.jpg','ATT169_19XX3_photo1-20190528-111338.jpg','ATT170_19XX3_photo2-20190528-111442.jpg','ATT171_17XX4_photo1-20190530-134332.jpg','ATT173_17XX2_photo1-20190530-104044.jpg','ATT174_17XX2_photo2-20190530-104919.jpg','ATT175_17XX2_photo3-20190530-104746.jpg','ATT176_19XX5_photo1-20190528-141335.jpg','ATT177_19XX5_photo2-20190528-141411.jpg','ATT178_19XX2_photo1-20190528-101401.jpg','ATT179_827X8_photo1-20190530-101718.jpg','ATT180_827X7_photo1-20190530-115040.jpg','ATT181_827X3_photo1-20190530-110410.jpg','ATT182_827X4_photo1-20190530-094525.jpg','ATT183_827X6_photo1_2019.jpg','ATT184_827X6_photo2_2019.jpg','ATT185_1121X2_2019_IMG_0033.jpg','ATT186_1121X2_2019_IMG_0035.jpg','ATT187_1121X2_2019_IMG_0036.jpg','ATT188_1121X2_2019_IMG_0038.jpg','ATT189_1121X3_2019_IMG_0023.jpg','ATT190_1121X3_2019_IMG_0024.jpg','ATT191_1121X3_2019_IMG_0025.jpg','ATT192_1121X3_2019_IMG_0026.jpg','ATT193_1121X4_2019_IMG_0020.jpg','ATT194_1121X4_2019_IMG_0021.jpg','ATT195_1121X4_2019_IMG_0022.jpg','ATT196_1121X5_2019_IMG_0016.jpg','ATT197_1121X5_2019_IMG_0017.jpg','ATT198_1121X5_2019_IMG_0018.jpg','ATT199_1121X5_2019_IMG_0019.jpg','ATT1_659X4_photo1-2019.jpg','ATT2019_626D1_2023_photo1-20230712-091037.jpg','ATT201_1155XX5_photo1-20190531-145744.jpg','ATT202_1155XX5_photo2-20190531-145915.jpg','ATT203_1155XX5_photo3-20190531-145944.jpg','ATT204_1155XX5_photo4-20190531-150020.jpg','ATT205_1155XX4_photo1-20190531-103648.jpg','ATT206_1155XX4_photo2-20190531-103736.jpg','ATT207_1155XX3_photo1-20190531-134948.jpg','ATT208_1155XX3_photo2-20190531-135149.jpg','ATT209_1155XX3_photo3-20190531-135440.jpg','ATT211_1155XX2_photo1-20190603-104805.jpg','ATT212_1155XX2_photo2-20190603-104900.jpg','ATT213_1155XX2_photo3-20190603-105043.jpg','ATT214_1155XX2_photo4-20190603-105145.jpg','ATT216_1155XX1_photo1-20190603-091459.jpg','ATT217_1155XX1_photo2-20190603-091546.jpg','ATT218_1155XX1_photo3-20190603-091640.jpg','ATT220_1175XX5_photo1-20190530-130810.jpg','ATT221_1175XX5_photo2-20190530-130912.jpg','ATT222_1175XX5_photo3-20190530-131109.jpg','ATT223_1175XX5_photo4-20190530-131214.jpg','ATT225_1175XX2_photo1-20190530-102115.jpg','ATT226_1175XX2_photo2-20190530-102159.jpg','ATT227_1175XX2_photo3-20190530-102357.jpg','ATT228_1175XX2_photo4-20190530-102718.jpg','ATT229_1175XX4_photo1-20190530-144419.jpg','ATT230_1175XX4_photo2-20190530-144509.jpg','ATT231_1175XX4_photo3-20190530-144539.jpg','ATT232_366XX3_photo1-20190523-111054.jpg','ATT233_366XX3_photo2-20190523-111117.jpg','ATT234_366XX1_photo1-20190523-133615.jpg','ATT235_366XX1_photo2-20190523-133652.jpg','ATT236_366XX5_photo1-20190522-152710.jpg','ATT237_366XX5_photo2-20190522-152734.jpg','ATT238_366XX4_photo1-20190522-120836.jpg','ATT239_366XX4_photo2-20190522-121038.jpg','ATT240_366XX2_photo1-20190522-125425.jpg','ATT241_366XX2_photo2-20190522-125454.jpg','ATT242_366XX2_photo3-20190522-125754.jpg','ATT243_478XX2_photo1-20190521-100928.jpg','ATT244_478XX2_photo2-20190521-101118.jpg','ATT245_478XX3_photo1-20190521-125226.jpg','ATT246_478XX3_photo2-20190521-125343.jpg','ATT247_15X5_photo1-20190605-100115.jpg','ATT248_15X1_photo1-20190604-141634.jpg','ATT249_15X1_photo2-20190604-141705.jpg','ATT250_15X4_photo1-20190603-143308.jpg','ATT251_15X4_photo2-20190603-143325.jpg','ATT252_15X3_photo1-20190603-161016.jpg','ATT253_15X3_photo2-20190603-161052.jpg','ATT254_15X2_photo1-20190605-114438.jpg','ATT255_15X2_photo2-20190605-114509.jpg','ATT256_1034XX5_photo1-20190606-113135.jpg','ATT257_1034XX5_photo2-20190606-114706.jpg','ATT258_1034XX5_photo3-20190606-114749.jpg','ATT259_1034XX4_photo1-20190605-143736.jpg','ATT260_1034XX4_photo2-20190605-143758.jpg','ATT261_1034XX3_photo1-20190605-160255.jpg','ATT262_1034XX3_photo2-20190605-160416.jpg','ATT266_1034XX2_photo1-20190606-145514.jpg','ATT267_1034XX2_photo2-20190606-145622.jpg','ATT268_1034XX2_photo3-20190606-145705.jpg','ATT271_1034XX1_photo1-20190606-133848.jpg','ATT272_1034XX1_photo2-20190606-133914.jpg','ATT273_1034XX1_photo3-20190606-134107.jpg','ATT274_125X5_photo1-20190611-103705.jpg','ATT275_125X5_photo2-20190611-103743.jpg','ATT276_99X3_photo1-20190606-154838.jpg','ATT277_99X3_photo2-20190606-154929.jpg','ATT278_125X4_photo1-20190611-115935.jpg','ATT279_125X4_photo2-20190611-120231.jpg','ATT280_125X3_photo1-20190610-152119.jpg','ATT281_125X3_photo2-20190610-152155.jpg','ATT282_125X1_photo1-20190610-131430.jpg','ATT283_125X1_photo2-20190610-131501.jpg','ATT284_723X4_photo1-20190610-121220.jpg','ATT285_723X4_photo2-20190610-121300.jpg','ATT286_723X4_photo3-20190610-121455.jpg','ATT287_723X5_photo1-20190610-091429.jpg','ATT288_723X5_photo2-20190610-091616.jpg','ATT289_723X5_photo3-20190610-091718.jpg','ATT290_723X5_photo4-20190610-091918.jpg','ATT292_723X3_photo1-20190611-090714.jpg','ATT293_723X3_photo2-20190611-090853.jpg','ATT294_723X1_photo1-20190611-111734.jpg','ATT295_723X1_photo2-20190611-111808.jpg','ATT296_723X1_photo3-20190611-111840.jpg','ATT297_723X1_photo4-20190611-112057.jpg','ATT298_723X2_photo1-20190610-141002.jpg','ATT299_723X2_photo2-20190610-141047.jpg','ATT2_659X4_photo2-2019.jpg','ATT300_125X2_photo1-20190610-142102.jpg','ATT301_125X2_photo2-20190610-142211.jpg','ATT302_99X4_photo1-20190606-170116.jpg','ATT303_99X4_photo2-20190606-170143.jpg','ATT304_1130X1_photo4-20190522-111704.jpg','ATT305_61XX1_photo1-20190619-134654.jpg','ATT306_61XX1_photo2-20190619-134714.jpg','ATT307_61XX1_photo3-20190619-134749.jpg','ATT309_89X5_photo1-20190618-131523.jpg','ATT310_89X5_photo2-20190618-131422.jpg','ATT311_89X1_photo1-20190618-083517.jpg','ATT312_89X1_photo2-20190618-083639.jpg','ATT313_89X2_photo1-20190618-105146.jpg','ATT314_89X2_photo2-20190618-105320.jpg','ATT315_89X3_photo1_2019.jpg','ATT316_89X3_photo2_2019.jpg','ATT317_89X3_photo3_2019.jpg','ATT319_61XX2_photo1-20190620-080450.jpg','ATT320_61XX2_photo2-20190620-080530.jpg','ATT321_61XX4_photo1-20190619-120541.jpg','ATT322_61XX4_photo2-20190619-120611.jpg','ATT323_61XX4_photo3-20190619-120643.jpg','ATT324_61XX5_photo1-20190619-111206.jpg','ATT325_61XX5_photo2-20190619-111301.jpg','ATT326_61XX5_photo3-20190619-111411.jpg','ATT327_99X5_photo1-20190606-095425.jpg','ATT328_99X1_photo1-20190606-120109.jpg','ATT329_99X1_photo2-20190606-120144.jpg','ATT330_101XX1_photo1-20190612-133036.jpg','ATT331_101XX1_photo2-20190612-133058.jpg','ATT332_101XX1_photo3-20190612-133119.jpg','ATT333_101XX4_photo1-20190612-103836.jpg','ATT334_101XX4_photo2-20190612-104027.jpg','ATT335_101XX3_photo1-20190612-094339.jpg','ATT336_101XX3_photo2-20190612-094410.jpg','ATT337_101XX3_photo3-20190612-094520.jpg','ATT338_101XX3_photo4-20190612-094558.jpg','ATT339_101XX5_photo1-20190612-144440.jpg','ATT340_101XX5_photo2-20190612-144505.jpg','ATT342_101XX2_photo1-20190612-114235.jpg','ATT343_101XX2_photo2-20190612-115657.jpg','ATT344_99X2_photo1_2019.jpg','ATT345_987X5_photo1-20190618-112950.jpg','ATT346_987X5_photo2-20190618-113130.jpg','ATT347_987X5_photo3-20190618-113416.jpg','ATT348_987X5_photo4-20190618-113915 from East corner toSW.jpg','ATT349_987X4_photo1-20190618-143857.jpg','ATT350_987X4_photo2-20190618-144044.jpg','ATT351_987X4_photo3-20190618-144121.jpg','ATT352_987X4_photo4-20190618-144153.jpg','ATT353_1056XX5_photo1-20190610-135419.jpg','ATT354_1056XX5_photo2-20190610-135443.jpg','ATT355_1056XX5_photo3-20190610-135508.jpg','ATT357_1056XX4_photo1-20190611-095919.jpg','ATT358_1056XX4_photo2-20190611-095946.jpg','ATT359_1056XX4_photo3-20190611-100027.jpg','ATT360_1056XX4_photo4-20190611-100134.jpg','ATT362_1056XX3_photo1-20190611-122939.jpg','ATT363_1056XX3_photo2-20190611-123008.jpg','ATT365_1056XX2_photo1-20190610-144703.jpg','ATT366_1056XX2_photo2-20190610-144810.jpg','ATT367_1056XX2_photo3-20190610-144851.jpg','ATT368_1056XX2_photo4-20190610-144946.jpg','ATT369_1056XX1_photo1-20190610-114310.jpg','ATT370_1056XX1_photo2-20190610-114552.jpg','ATT371_1056XX1_photo3-20190610-111633.jpg','ATT372_1056XX1_photo4-20190610-114202.jpg','ATT373_982X5_photo1-20190613-094229.jpg','ATT374_982X5_photo2-20190613-094340.jpg','ATT375_982X5_photo3-20190613-094446.jpg','ATT376_982X4_photo1-20190612-141636.jpg','ATT377_982X4_photo2-20190612-141716.jpg','ATT378_982X3_photo1-20190612-105558.jpg','ATT379_982X3_photo2-20190612-105042.jpg','ATT380_982X3_photo3-20190612-105145.jpg','ATT381_982X3_photo4-20190612-105206.jpg','ATT382_982X2_photo1-20190614-105846 from South to Centre.jpg','ATT383_982X2_photo2-20190614-110031.jpg','ATT384_982X2_photo3-20190614-110120.jpg','ATT385_982X2_photo4-20190614-110217.jpg','ATT386_982X1_photo1-20190614-093519 to NWest.jpg','ATT387_982X1_photo2-20190614-091209 from West to East.jpg','ATT388_982X1_photo3-20190614-093549 To NorthEAST.jpg','ATT389_83XX3_photo1-20190624-122122.jpg','ATT390_83XX3_photo2-20190624-122200.jpg','ATT391_83XX3_photo3-20190624-122232.jpg','ATT392_83XX3_photo4-20190624-122300.jpg','ATT394_83XX2_photo1-20190624-134232.jpg','ATT395_83XX2_photo2-20190624-134307.jpg','ATT396_83XX2_photo3-20190624-134357.jpg','ATT398_83XX5_photo1-20190625-101546.jpg','ATT399_83XX5_photo2-20190625-101611.jpg','ATT3_659X4_photo3-2019.jpg','ATT400_83XX1_photo1-20190625-085042.jpg','ATT401_83XX1_photo2-20190625-085101.jpg','ATT402_703XX5_photo1-20190625-092556.jpg','ATT403_703XX5_photo2-20190625-092626.jpg','ATT404_703XX5_photo3-20190625-092653.jpg','ATT405_703XX5_photo4-20190625-092716.jpg','ATT407_703XX4_photo1-20190625-105959.jpg','ATT408_703XX4_photo2-20190625-110035.jpg','ATT409_703XX4_photo3-20190625-110110.jpg','ATT410_703XX4_photo4-20190625-110140.jpg','ATT411_703XX3_photo1-20190624-120749.jpg','ATT412_703XX3_photo2-20190624-120916.jpg','ATT413_703XX3_photo3-20190624-121010.jpg','ATT415_703XX2_photo1-20190624-132533.jpg','ATT416_703XX2_photo2-20190624-132558.jpg','ATT417_703XX2_photo3-20190624-132619.jpg','ATT418_703XX2_photo4-20190624-132640.jpg','ATT420_703XX7_photo1-20190624-100614.jpg','ATT421_703XX7_photo2-20190624-100640.jpg','ATT422_703XX7_photo3-20190624-100706.jpg','ATT423_703XX7_photo4-20190624-100723.jpg','ATT424_184XX1_photo1-20190626-131851.jpg','ATT425_184XX1_photo2-20190626-134717.jpg','ATT426_184XX1_photo3-20190626-134849.jpg','ATT427_184XX1_photo4-20190626-134918.jpg','ATT428_184XX5_photo1-20190626-142630.jpg','ATT429_184XX5_photo2-20190626-142741.jpg','ATT430_184XX5_photo3-20190626-142825.jpg','ATT432_184XX2_photo1_2019.jpg','ATT433_184XX2_photo2_2019.jpg','ATT434_184XX2_photo3_2019.jpg','ATT435_184XX2_photo4_2019.jpg','ATT436_910XX5_photo1-20190626-095422.jpg','ATT437_910XX5_photo2-20190626-095449.jpg','ATT438_910XX3_photo1-20190626-110614.jpg','ATT439_910XX3_photo2-20190626-110646.jpg','ATT440_910XX2_photo1-20190626-114920.jpg','ATT441_910XX2_photo2-20190626-114942.jpg','ATT442_910XX6_photo1-20190625-115156.jpg','ATT443_910XX6_photo2-20190625-115217.jpg','ATT444_910XX1_photo1-20190625-104635.jpg','ATT445_910XX1_photo2-20190625-104655.jpg','ATT446_910XX1_photo3-20190625-104719.jpg','ATT447_861XX5_photo1-20190619-111142.jpg','ATT448_861XX5_photo2-20190619-111214.jpg','ATT449_861XX2_photo1-20190619-095416.jpg','ATT450_861XX2_photo2-20190619-095547.jpg','ATT451_861XX1_photo1-20190619-090238.jpg','ATT452_861XX1_photo2-20190619-090307.jpg','ATT454_861XX6_photo1-20190618-105516.jpg','ATT455_861XX6_photo2-20190618-105532.jpg','ATT456_861XX3_photo1-20190618-095848.jpg','ATT457_861XX3_photo2-20190618-095915.jpg','ATT458_861XX3_photo3-20190618-095948.jpg','ATT459_898XX5_photo1-20190611-110626.jpg','ATT460_898XX5_photo2-20190611-110653.jpg','ATT461_898XX2_photo1-20190611-102137.jpg','ATT462_898XX2_photo2-20190611-102203.jpg','ATT463_837X5_photo1-20190701-141846.jpg','ATT464_837X5_photo2-20190701-141944.jpg','ATT465_837X5_photo3-20190701-142016.jpg','ATT466_837X4_photo1-20190701-144407.jpg','ATT467_837X4_photo2-20190701-144423.jpg','ATT468_837X4_photo3-20190701-144441.jpg','ATT469_951X5_photo1-20190627-092112.jpg','ATT470_951X5_photo2-20190627-092152.jpg','ATT471_951X5_photo3-20190627-092216.jpg','ATT472_951X4_photo1-20190627-132945.jpg','ATT473_951X4_photo2-20190627-133002.jpg','ATT474_951X3_photo1-20190628-110648.jpg','ATT475_951X3_photo2-20190628-110738.jpg','ATT476_951X3_photo3-20190628-110831.jpg','ATT477_951X2_photo1-20190628-092927.jpg','ATT478_951X2_photo2-20190628-092949.jpg','ATT479_951X2_photo3-20190628-093017.jpg','ATT480_951X1_photo1-20190627-105700.jpg','ATT481_951X1_photo2-20190627-105733.jpg','ATT482_987X3_photo1-20190625-122141.jpg','ATT483_987X3_photo2-20190625-122214.jpg','ATT484_987X3_photo3-20190625-122310.jpg','ATT485_987X3_photo4-20190625-122348.jpg','ATT486_987X2_photo1-20190625-155642.jpg','ATT487_987X2_photo2-20190625-155713.jpg','ATT488_987X2_photo3-20190625-155754.jpg','ATT489_987X1_photo1-20190625-150545.jpg','ATT490_987X1_photo2-20190625-150612.jpg','ATT491_987X1_photo3-20190625-150637.jpg','ATT492_955X1_photo1-20190624-160007.jpg','ATT493_955X1_photo2-20190624-160042.jpg','ATT494_955X1_photo3-20190624-160118.jpg','ATT495_955X1_photo4-20190624-160200.jpg','ATT496_955X5_photo1-20190619-141712.jpg','ATT497_955X5_photo2-20190619-141825.jpg','ATT498_955X5_photo3-20190619-141856.jpg','ATT499_955X5_photo4-20190619-142020.jpg','ATT500_955X4_photo1-20190619-105418.jpg','ATT501_955X4_photo2-20190619-105717.jpg','ATT502_955X4_photo3-20190619-110003.jpg','ATT503_955X4_photo4-20190619-110045.jpg','ATT504_955X3_photo1-20190624-133132.jpg','ATT505_955X3_photo2-20190624-133200.jpg','ATT506_955X3_photo3-20190624-133239.jpg','ATT507_955X2_photo1-20190624-151028.jpg','ATT508_955X2_photo2-20190624-151052.jpg','ATT509_955X2_photo3-20190624-151117.jpg','ATT510_817X3_photo1-20190702-110729.jpg','ATT511_817X3_photo2-20190702-110750.jpg','ATT512_817X3_photo3-20190702-110829.jpg','ATT513_837X2_photo1-20190703-121017.jpg','ATT514_837X2_photo2-20190703-121133.jpg','ATT515_837X1_photo1-20190703-100756.jpg','ATT516_837X1_photo2-20190703-100914.jpg','ATT517_837X1_photo3-20190703-101013.jpg','ATT518_837X1_photo4-20190703-101043.jpg','ATT519_837X3_photo1-20190703-085250.jpg','ATT520_837X3_photo2-20190703-085345.jpg','ATT521_837X3_photo3-20190703-085421.jpg','ATT522_817X2_photo1-20190702-121227.jpg','ATT523_817X2_photo2-20190702-121245.jpg','ATT524_817X5_photo1-20190702-091738.jpg','ATT525_817X5_photo2-20190702-091802.jpg','ATT526_937X4_photo1-20190709-103552.jpg','ATT527_937X4_photo2-20190709-103732.jpg','ATT528_937X5_photo1-20190709-131722.jpg','ATT529_937X5_photo2-20190709-131746.jpg','ATT530_937X3_photo1-20190708-134042.jpg','ATT531_937X3_photo2-20190708-134123.jpg','ATT532_937X3_photo3-20190708-134238.jpg','ATT533_937X3_photo4-20190708-134352.jpg','ATT534_937X2_photo1-20190708-104528.jpg','ATT535_937X2_photo2-20190708-104656.jpg','ATT536_937X2_photo3-20190708-104812.jpg','ATT537_937X1_photo1-20190708-115926.jpg','ATT538_937X1_photo2-20190708-120150.jpg','ATT539_937X1_photo3-20190708-120300.jpg','ATT540_937X1_photo4-20190708-120441.jpg','ATT541_937X4_2019_IMG_3158_X4.JPG','ATT542_214XX4_photo1-20190708-131106.jpg','ATT543_214XX4_photo2-20190708-131849.jpg','ATT544_214XX4_photo3-20190708-132551.jpg','ATT546_214XX1_photo1_2019_198f424d8edb4e31adeb7b5dba975048.jpg','ATT547_214XX1_photo2-20190708-095015.jpg','ATT550_214XX5_photo1-20190708-122651.jpg','ATT551_214XX5_photo2-20190708-122743.jpg','ATT552_214XX5_photo3-20190708-122812.jpg','ATT555_300XX3_photo1-20190709-125919.jpg','ATT556_300XX3_photo2-20190709-125959.jpg','ATT558_300XX1_photo1-20190709-100722.jpg','ATT559_300XX1_photo2-20190709-102341.jpg','ATT560_300XX1_photo3-20190709-102503.jpg','ATT562_214XX2_photo1-20190708-104733.jpg','ATT563_214XX2_photo3-20190708-110200.jpg','ATT564_214XX2_photo4-20190708-110101.jpg','ATT565_300XX5_photo1-20190709-143336.jpg','ATT566_300XX5_photo2-20190709-143410.jpg','ATT567_300XX4_photo1-20190709-133818.jpg','ATT568_300XX4_photo2-20190709-133835.jpg','ATT569_300XX2_photo1-20190709-113028.jpg','ATT570_300XX2_photo2-20190709-113116.jpg','ATT571_214XX3_photo1-20190708-135825.jpg','ATT572_214XX3_photo2-20190708-135855.jpg','ATT573_214XX3_photo3-20190708-140014.jpg','ATT574_661XX3_photo1-20190711-095344.jpg','ATT575_661XX3_photo2-20190711-095406.jpg','ATT576_661XX3_photo3-20190711-095735.jpg','ATT577_661XX4_photo1-20190711-085233.jpg','ATT578_661XX4_photo2-20190711-085302.jpg','ATT579_661XX4_photo3-20190711-085354.jpg','ATT580_661XX4_photo4-20190711-085408.jpg','ATT581_661XX5_photo1-20190710-131225.jpg','ATT582_661XX5_photo2-20190710-131249.jpg','ATT583_661XX5_photo3-20190710-131304.jpg','ATT584_661XX5_photo4-20190710-131319.jpg','ATT585_661XX2_photo1-20190710-115221.jpg','ATT586_661XX2_photo2-20190710-115254.jpg','ATT587_661XX2_photo3-20190710-115329.jpg','ATT588_661XX2_photo4-20190710-115352.jpg','ATT589_661XX1_photo1-20190710-102022.jpg','ATT590_661XX1_photo2-20190710-102049.jpg','ATT591_661XX1_photo3-20190710-102107.jpg','ATT592_661XX1_photo4-20190710-102126.jpg','ATT593_266XX3_photo1-20190715-154247.jpg','ATT594_266XX3_photo2-20190715-154431.jpg','ATT596_266XX1_photo1-20190715-125405.jpg','ATT597_266XX1_photo2-20190715-125450.jpg','ATT598_266XX1_photo3-20190715-125520.jpg','ATT599_266XX1_photo4-20190715-131415.jpg','ATT59_658X1_photo1-20190515-140737.jpg','ATT600_266XX2_photo1-20190715-134954.jpg','ATT601_266XX2_photo2-20190715-135126.jpg','ATT602_266XX2_photo3-20190715-140456.jpg','ATT603_266XX4_photo1-20190715-112043.jpg','ATT604_266XX4_photo2-20190715-112424.jpg','ATT605_266XX4_photo3-20190715-112824.jpg','ATT607_266XX5_photo1-20190716-085403.jpg','ATT608_266XX5_photo2-20190716-085459.jpg','ATT609_266XX5_photo3-20190716-085536.jpg','ATT60_658X1_photo2-20190515-140817.jpg','ATT610_266XX5_photo4-20190716-085644.jpg','ATT612_637XX5_photo1-20190715-132108.jpg','ATT613_637XX5_photo2-20190715-132129.jpg','ATT614_637XX5_photo3-20190715-132222.jpg','ATT615_637XX3_photo1-20190716-104528.jpg','ATT616_637XX3_photo2-20190716-104552.jpg','ATT617_637XX1_photo1-20190716-092338.jpg','ATT618_637XX1_photo2-20190716-092404.jpg','ATT619_637XX2_photo1-20190716-082749.jpg','ATT620_637XX2_photo2-20190716-082814.jpg','ATT622_637XX4_photo1-20190715-110726.jpg','ATT623_637XX4_photo2-20190715-110759.jpg','ATT624_1104X4_photo1-20190717-151938.jpg','ATT625_1104X4_photo2-20190717-152043.jpg','ATT626_1104X3_photo1-20190717-140809.jpg','ATT627_1104X3_photo2-20190717-140920.jpg','ATT62_658X4_photo1-20190515-141608 from north.jpg','ATT630_774X7_photo1-20190703-110635.jpg','ATT631_774X4_photo1-20190703-101055.jpg','ATT632_774X4_photo2-20190703-101147.jpg','ATT633_774X6_photo1-20190702-132112.jpg','ATT634_774X6_photo2-20190702-132144.jpg','ATT636_774X3_photo1-20190702-120834.jpg','ATT637_774X3_photo2-20190702-120905.jpg','ATT638_774X8_photo1-20190702-105211.jpg','ATT639_774X8_photo2-20190702-105247.jpg','ATT63_658X4_photo2-20190515-141713 from west.jpg','ATT640_1104X5_photo1-20190717-151052.jpg','ATT641_1104X5_photo2-20190717-151114.jpg','ATT642_1104X1_photo1-20190717-140617.jpg','ATT643_1104X2_photo1-20190717-163955.jpg','ATT644_1104X1_IMG_2044_2019.jpg','ATT645_1104X1_IMG_2045_2019.jpg','ATT646_1104X1_IMG_2046_2019.jpg','ATT647_1104X1_IMG_2048_2019.jpg','ATT648_1104X5_IMG_2053_2019.jpg','ATT649_1104X5_IMG_2054_2019.jpg','ATT64_658X4_photo3-20190515-141745 from south.jpg','ATT650_1104X5_IMG_2055_2019.jpg','ATT651_1104X2_IMG_2056_2019.jpg','ATT652_1104X2_IMG_2057_2019.jpg','ATT653_1084X2_photo1-20190718-121316.jpg','ATT654_1084X2_photo2-20190718-121335.jpg','ATT655_1084X2_photo3-20190718-121433.jpg','ATT656_1084X2_photo4-20190718-121448towardscorncrake_South.jpg','ATT657_1084X1_photo1-20190718-092900.jpg','ATT658_1084X1_photo2-20190718-092924.jpg','ATT659_1084X1_photo3-20190718-092956.jpg','ATT65_658X4_photo4-20190515-141811 from east.jpg','ATT660_231XX1_photo1-20190722-144755.jpg','ATT661_231XX1_photo2_2019.jpg','ATT662_231XX1_photo3_2019.jpg','ATT663_231XX2_photo1-20190722-132901.jpg','ATT664_231XX2_photo2-20190722-132927.jpg','ATT665_231XX2_photo3_2019.jpg','ATT666_231XX4_photo1-20190723-084012.jpg','ATT667_231XX4_photo2-20190723-084043.jpg','ATT668_260XX3_photo1-20190724-124258.jpg','ATT669_260XX3_photo2-20190724-124350.jpg','ATT670_260XX4_photo1-20190724-150634.jpg','ATT671_260XX4_photo2-20190724-150717.jpg','ATT672_260XX4_photo3-20190724-150748.jpg','ATT673_260XX2_photo1-20190724-113718.jpg','ATT674_260XX2_photo2-20190724-113758.jpg','ATT675_260XX2_photo3_2019.jpg','ATT676_260XX5_photo1-20190724-135536.jpg','ATT677_260XX5_photo2-20190724-135622.jpg','ATT678_260XX1_photo1-20190724-100805.jpg','ATT679_260XX1_photo2-20190724-100846.jpg','ATT67_658X5_photo1-20190515-144112.jpg','ATT680_260XX1_photo3_2019.jpg','ATT681_231XX5_photo1-20190723-094913.jpg','ATT682_231XX5_photo2-20190723-094949.jpg','ATT683_400X1_photo1-20190723-131646.jpg','ATT684_400X1_photo2-20190723-131730.jpg','ATT685_447X4_photo1-20190715-125628.jpg','ATT686_447X4_photo2-20190715-125649.jpg','ATT687_451X1_photo1-20190722-143357.jpg','ATT688_400X4_photo1-20190724-095254.jpg','ATT689_400X4_photo2-20190724-104243.jpg','ATT68_658X5_photo2-20190515-144220.jpg','ATT690_400X4_photo3-20190724-104312.jpg','ATT691_400X4_photo4-20190724-104434.jpg','ATT692_451X5_photo1-20190722-124837.jpg','ATT693_451X5_photo2-20190722-124910.jpg','ATT694_400X2_photo1-20190723-105557.jpg','ATT695_400X2_photo2-20190723-105628.jpg','ATT696_400X3_photo1-20190723-150915.jpg','ATT697_400X3_photo2-20190723-150948.jpg','ATT698_447X3_photo1-20190716-132318.jpg','ATT699_447X1_photo1-20190716-112528.jpg','ATT69_658X5_photo3-20190515-144311.jpg','ATT700_447X1_photo2-20190716-112629.jpg','ATT701_447X5_photo1-20190716-100623.jpg','ATT702_447X5_photo2-20190716-101009.jpg','ATT703_447X2_photo1-20190715-150643.jpg','ATT704_447X2_photo2-20190715-150705.jpg','ATT705_254XX5_photo1_2019.jpg','ATT706_254XX5_photo2_2019.jpg','ATT707_254XX5_photo3_2019.jpg','ATT708_254XX4_photo1_2019.jpg','ATT709_254XX4_photo2-20190729-131230.jpg','ATT70_658X2_photo1-20190515-120144.jpg','ATT710_254XX4_photo3_2019.jpg','ATT711_231XX3_photo1_2019.jpg','ATT712_231XX3_photo2_2019.jpg','ATT713_316XX4_photo1-20190730-105959.jpg','ATT714_316XX4_photo2-20190730-110040.jpg','ATT715_316XX4_photo3-20190730-110248.jpg','ATT716_316XX5_photo1-20190730-093622.jpg','ATT717_316XX5_photo2-20190730-093641.jpg','ATT718_757XX1_photo1-20190808-114054.jpg','ATT719_757XX1_photo2-20190808-114116.jpg','ATT71_658X2_photo2-20190515-120340.jpg','ATT720_757XX1_photo3-20190808-114134.jpg','ATT721_757XX1_photo4-20190808-114157.jpg','ATT722_757XX2_photo1-20190808-101810.jpg','ATT723_757XX2_photo2-20190808-101841.jpg','ATT724_757XX2_photo3-20190808-101944.jpg','ATT725_757XX2_photo4-20190808-102014.jpg','ATT726_757XX3_photo1-20190807-133423.jpg','ATT727_757XX3_photo2-20190807-133438.jpg','ATT728_757XX3_photo3-20190807-133501.jpg','ATT729_757XX3_photo4-20190807-133526.jpg','ATT730_757XX5_photo1-20190807-121022.jpg','ATT731_757XX5_photo2-20190807-121046.jpg','ATT732_757XX5_photo3-20190807-121105.jpg','ATT733_757XX5_photo4-20190807-121123.jpg','ATT734_757XX4_photo1-20190807-103307.jpg','ATT735_757XX4_photo2-20190807-103325.jpg','ATT736_757XX4_photo3-20190807-103352.jpg','ATT737_757XX4_photo4-20190807-103416.jpg','ATT738_743X3_photo1-20190806-103603.jpg','ATT739_743X3_photo2-20190806-103620.jpg','ATT73_658X2_photo4-20190515-121751.jpg','ATT740_743X3_photo3-20190806-103640.jpg','ATT741_743X3_photo4-20190806-103711.jpg','ATT742_743X1_photo1-20190806-090417.jpg','ATT743_743X1_photo2-20190806-090436.jpg','ATT744_743X1_photo3-20190806-090459.jpg','ATT745_743X1_photo4-20190806-090537.jpg','ATT746_743X5_photo1-20190805-114232.jpg','ATT747_743X5_photo2-20190805-114255.jpg','ATT748_743X5_photo3-20190805-114317.jpg','ATT749_743X5_photo4-20190805-114334.jpg','ATT74_1213X1_photo1-20190516-134945.jpg','ATT751_743X2_photo1-20190805-102633.jpg','ATT752_743X2_photo2-20190805-102737.jpg','ATT753_743X2_photo3-20190805-102813.jpg','ATT754_743X2_photo4-20190805-102849.jpg','ATT755_743X4_photo1-20190805-124454.jpg','ATT756_743X4_photo2-20190805-124511.jpg','ATT757_743X4_photo3-20190805-124541.jpg','ATT758_743X4_photo4-20190805-124559.jpg','ATT759_835X5_photo1-20190813-105752_SOUTHPoint.jpg','ATT75_1213X1_photo2-20190516-135013.jpg','ATT760_835X5_photo2-20190813-110013_SOUTHpoint.jpg','ATT761_835X5_photo3-20190813-110055_SOUTHpoint.jpg','ATT762_835X2_photo1-20190813-095023.jpg','ATT764_835X1_photo1-20190813-123653.jpg','ATT765_835X1_photo2-20190813-123755.jpg','ATT766_835X1_photo3-20190813-123900.jpg','ATT767_835X3_photo1-20190814-112952.jpg','ATT768_835X3_photo2-20190814-113033.jpg','ATT769_835X3_photo3-20190814-113630.jpg','ATT76_1213X3_photo1-20190516-150852.jpg','ATT770_835X3_photo4-20190814-114640_BEECHfromsketchmap.jpg','ATT774_932XX2_photo1-20190808-151747.jpg','ATT775_932XX3_photo1-20190808-162148.jpg','ATT776_932XX3_photo2-20190808-162239.jpg','ATT777_932XX3_photo3-20190808-162326.jpg','ATT778_932XX3_photo4-20190808-162411.jpg','ATT787_1020X5_IMG_2151_2019.jpg','ATT788_1020X5_IMG_2152_2019.jpg','ATT789_1020X5_IMG_2153_2019.jpg','ATT78_1213X2_photo1-20190516-122556.jpg','ATT790_1020X5_IMG_2154_2019.jpg','ATT791_796X5_photo1-20190819-125144.jpg','ATT792_796X5_photo2-20190819-125452.jpg','ATT793_796X5_photo3-20190819-125313.jpg','ATT794_796X5_photo4-20190819-125411.jpg','ATT795_796X4_photo1-20190818-141633.jpg','ATT796_796X4_photo2-20190818-141721.jpg','ATT797_796X4_photo3-20190818-142255.jpg','ATT798_796X4_photo4-20190818-141853JenAsTheMissingTelPoleFrom98.jpg','ATT799_796X3_photo1-20190818-154701.jpg','ATT79_1213X2_photo2-20190516-135918.jpg','ATT800_796X3_photo2-20190818-154750.jpg','ATT801_796X3_photo3-20190818-154826.jpg','ATT802_796X3_photo4-20190818-154901.jpg','ATT803_796X2_photo1-20190819-111140.jpg','ATT804_796X2_photo2-20190819-111210.jpg','ATT805_796X2_photo3-20190819-111300.jpg','ATT806_796X2_photo4-20190819-111335.jpg','ATT807_796X1_photo1-20190819-094150.jpg','ATT808_796X1_photo2-20190819-094311.jpg','ATT809_796X1_photo3-20190819-094401.jpg','ATT80_1213X2_photo3-20190516-140008.jpg','ATT810_68X3_photo1-20190813-133725.jpg','ATT811_68X1_photo1-20190813-155536.jpg','ATT812_68X4_photo1-20190813-110805.jpg','ATT813_68X2_photo2-20190813-110826.jpg','ATT814_68X5_photo1-20190812-135628.jpg','ATT815_68X4_photo1-20190812-160320.jpg','ATT816_341XX1_photo1-20190822-094858.jpg','ATT817_341XX1_photo2-20190822-094938.jpg','ATT818_341XX2_photo1-20190821-151631.jpg','ATT819_341XX2_photo2-20190821-151707.jpg','ATT81_1213X2_photo4-20190516-140052.jpg','ATT820_341XX2_photo3-20190821-152030.jpg','ATT821_341XX3_photo1-20190821-141955.jpg','ATT822_341XX4_photo1-20190821-132152.jpg','ATT823_341XX5_photo1-20190821-114329.jpg','ATT824_191X5_photo1-20190820-104453.jpg','ATT825_191X1_photo1-20190820-145822.jpg','ATT826_191X1_photo2-20190820-145933.jpg','ATT827_191X2_photo1-20190820-131739.jpg','ATT828_191X4_photo1-20190819-143259.jpg','ATT829_191X3_photo1-20190819-115535.jpg','ATT82_1213X1_photo1-20190516-145813.jpg','ATT830_90XX5_photo1-20190815-105112.jpg','ATT831_90XX3_photo1-20190814-132713.jpg','ATT833_90XX4_photo1-20190815-130902.jpg','ATT834_90XX4_photo2-20190815-130940.jpg','ATT835_90XX2_photo1-20190814-105906.jpg','ATT836_359X5_photo1-20190808-115012.jpg','ATT837_359X5_photo2-20190808-115112.jpg','ATT838_359X2_photo1-20190808-100819.jpg','ATT839_359X2_photo2-20190808-100851.jpg','ATT83_1213X1_photo2-20190516-145836.jpg','ATT840_359X2_photo3-20190808-100934.jpg','ATT841_359X4_photo1-20190807-152945.jpg','ATT842_359X1_photo1-20190807-133049.jpg','ATT843_359X1_photo2-20190807-133159.jpg','ATT844_359X3_photo1-20190807-105328.jpg','ATT845_359X3_photo2-20190807-105411.jpg','ATT846_471X3_photo1-20190806-133152.jpg','ATT848_471X4_photo1-20190806-094533.jpg','ATT849_471X4_photo2-20190806-094623.jpg','ATT84_1213X1_photo3-20190516-145909.jpg','ATT850_471X5_photo1-20190805-154023.jpg','ATT851_471X5_photo2-20190805-154147.jpg','ATT853_471X2_photo1-20190805-124412.jpg','ATT854_471X2_photo2-20190805-124511.jpg','ATT855_471X2_photo3-20190805-124537.jpg','ATT856_950XX5_photo1-20190807-110638.jpg','ATT857_950XX5_photo2-20190807-110749.jpg','ATT858_950XX1_photo1-20190807-121649.jpg','ATT859_950XX1_photo2-20190807-121718.jpg','ATT85_658X3_photo1-20190515-143652.jpg','ATT860_950XX4_photo1-20190807-093233.jpg','ATT861_950XX4_photo2-20190807-093303.jpg','ATT862_950XX3_photo1-20190807-141948.jpg','ATT863_950XX3_photo2-20190807-142010.jpg','ATT864_950XX2_photo1-20190807-131832.jpg','ATT865_950XX2_photo2-20190807-131746.jpg','ATT866_950XX2_photo3-20190807-131917.jpg','ATT867_878XX1_photo1-20190813-104841.jpg','ATT868_878XX4_photo1-20190813-125150.jpg','ATT869_878XX4_photo2-20190813-125231.jpg','ATT86_658X3_photo2-20190515-143731.jpg','ATT870_878XX3_photo1-20190813-154006.jpg','ATT871_878XX3_photo2-20190813-154051.jpg','ATT872_878XX5_photo1-20190813-132746.jpg','ATT873_878XX5_photo2-20190813-132806.jpg','ATT875_878XX6_photo1-20190813-144412.jpg','ATT876_878XX6_photo2-20190813-144436.jpg','ATT877_878XX6_photo3-20190813-144459.jpg','ATT878_931X5_photo1-20190806-074737.jpg','ATT879_931X5_photo2-20190806-074824.jpg','ATT87_658X3_photo3-20190515-143756.jpg','ATT880_931X3_photo1-20190806-093441.jpg','ATT881_931X3_photo2-20190806-093615.jpg','ATT882_931X4_photo1-20190806-112220.jpg','ATT883_931X4_photo2-20190806-112249.jpg','ATT884_751X5_photo1-20190821-092800.jpg','ATT885_751X5_photo2-20190821-092817.jpg','ATT886_751X5_photo3-20190821-092843.jpg','ATT887_751X4_photo1-20190821-115859.jpg','ATT888_751X4_photo2-20190821-115922.jpg','ATT889_751X3_photo1-20190821-125725.jpg','ATT88_658X3_photo4-20190515-143917.jpg','ATT890_751X3_photo2-20190821-125804.jpg','ATT891_751X3_photo3-20190821-125829.jpg','ATT892_751X2_photo1-20190821-103858.jpg','ATT893_751X2_photo2-20190821-103923.jpg','ATT894_751X2_photo3-20190821-103939.jpg','ATT895_751X1_photo1-20190821-135816.jpg','ATT896_751X1_photo2-20190821-135838.jpg','ATT897_765X1_photo1-20190820-112807.jpg','ATT898_765X1_photo2-20190820-112839.jpg','ATT900_765X6_photo1-20190820-163104.jpg','ATT901_765X6_photo2-20190820-163123.jpg','ATT902_765X3_photo1-20190820-152937.jpg','ATT903_765X3_photo2-20190820-153022.jpg','ATT905_765X4_photo1-20190820-142038.jpg','ATT906_765X4_photo2-20190820-142111.jpg','ATT907_765X2_photo1-20190820-125056.jpg','ATT908_765X2_photo2-20190820-125139.jpg','ATT909_1041X1_photo1_2019.jpg','ATT910_1041X1_photo2_2019.jpg','ATT911_1041X2_photo1_2019.jpg','ATT912_1041X2_photo2_2019.jpg','ATT913_1041X4_photo1_2019.jpg','ATT914_1041X4_photo2_2019.jpg','ATT915_983X5_photo1_2019.jpg','ATT916_983X5_photo2_2019.jpg','ATT917_983X5_photo3_2019.jpg','ATT919_983X6_photo1_2019.jpg','ATT920_983X6_photo2_2019.jpg','ATT922_983X1_photo1d_2019.jpg','ATT923_983X1_photo2_2019.jpg','ATT924_983X1_photo3a_2019.jpg','ATT925_1020X2_photo1-20190719-110039.jpg','ATT926_1020X2_photo2-20190719-110115.jpg','ATT927_1020X3_photo1-20190719-123134.jpg','ATT928_1020X3_photo2-20190719-123211.jpg','ATT929_1020X1_photo1-20190719-133910.jpg','ATT92_600X4_photo1-20190520-102323.jpg','ATT930_1020X1_photo2-20190719-133938.jpg','ATT931_1020X1_photo3-20190719-134038.jpg','ATT932_1084X5_photo1-20190718-094011.jpg','ATT933_1084X5_photo2-20190718-094050.jpg','ATT934_1084XX4_photo1-20190718-104931.jpg','ATT935_1084XX4_photo2-20190718-105006.jpg','ATT936_1084XX4_photo3-20190718-105036.jpg','ATT937_1084X3_photo1-20190718-130813.jpg','ATT938_1084X3_photo2-20190718-130838.jpg','ATT939_328X5_photo1a_2019.jpg','ATT93_600X4_photo2-20190520-110816.jpg','ATT940_328X5_photo2_2019.jpg','ATT941_328X4_photo1-20190819-142542.jpg','ATT942_328X4_photo2-20190819-142620.jpg','ATT943_381XX4_photo1-20190821-140420.jpg','ATT944_381XX4_photo2-20190821-140500.jpg','ATT945_328X3_photo1_2019.jpg','ATT946_328X3_photo2_2019.jpg','ATT947_328X2_photo1_2019.jpg','ATT948_328X2_photo2_2019.jpg','ATT949_328X2_photo3_2019.jpg','ATT94_600X4_photo3-20190520-102754.jpg','ATT950_328X1_photo1_2019.jpg','ATT951_328X1_photo2_2019.jpg','ATT952_328X1_photo3_2019.jpg','ATT953_381XX5_photo1-20190821-124352.jpg','ATT954_381XX5_photo2-20190821-124423.jpg','ATT955_381XX5_photo3-20190821-124517.jpg','ATT956_381XX3_photo1_2019.jpg','ATT957_381XX3_photo2_2019.jpg','ATT958_381XX3_photo3_2019.jpg','ATT959_381XX2_photo1-20190821-113009.jpg','ATT95_600X4_photo4-20190520-102844.jpg','ATT960_381XX2_photo2-20190821-113042.jpg','ATT961_381XX1_photo1_2019.jpg','ATT962_381XX1_photo2_2019.jpg','ATT963_543X1_photo1-20190827-140147.jpg','ATT964_543X1_photo2-20190827-140218.jpg','ATT965_543X1_photo3-20190827-140312.jpg','ATT966_543X3_photo1-20190827-162019.jpg','ATT967_543X3_photo2-20190827-162106.jpg','ATT968_543X2_photo1-20190827-150130.jpg','ATT969_543X2_photo2-20190827-150221.jpg','ATT970_541XX5_photo1-20190828-104255.jpg','ATT971_541XX5_photo2-20190828-104322.jpg','ATT972_541XX4_photo1-20190828-115259.jpg','ATT973_541XX4_photo2-20190828-115316.jpg','ATT974_541XX4_photo3-20190828-115344.jpg','ATT975_541XX3_photo1-20190828-131819.jpg','ATT976_541XX3_photo2-20190828-131841.jpg','ATT977_541XX2_photo1-20190828-140420.jpg','ATT978_541XX2_photo2-20190828-140448.jpg','ATT979_541XX1_photo1_2019.jpg','ATT97_600X5_photo1-20190520-123235.jpg','ATT980_541XX1_photo2_2019.jpg','ATT981_721XX3_photo1-20190820-101442.jpg','ATT982_721XX3_photo2-20190820-101456.jpg','ATT983_721XX3_photo3-20190820-101633.jpg','ATT984_721XX3_photo4-20190820-101650.jpg','ATT985_744XX5_photo1-20190821-090307.jpg','ATT986_744XX5_photo2-20190821-090340.jpg','ATT987_744XX5_photo3-20190821-090426.jpg','ATT988_744XX5_photo4-20190821-090454.jpg','ATT989_744XX4_photo1-20190821-095343.jpg','ATT98_600X5_photo2-20190520-124136.jpg','ATT990_744XX4_photo2-20190821-095446.jpg','ATT991_744XX4_photo3-20190821-095513.jpg','ATT992_744XX4_photo4-20190821-095644.jpg','ATT993_744XX1_photo1-20190821-122130.jpg','ATT994_744XX1_photo2-20190821-122205.jpg','ATT995_744XX1_photo3-20190821-122245.jpg','ATT996_744XX1_photo4-20190821-122312.jpg','ATT997_744XX3_photo1-20190821-133036.jpg','ATT998_744XX3_photo2-20190821-133108.jpg','ATT999_744XX3_photo3-20190821-133133.jpg','ATT99_600X5_photo3-20190520-124205.jpg']
    sqs = list()
    for s in cs_files:
        sqs += [numbers_in_str(s, type_=int)[1]]
    pass