# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument
"""rename files
"""

import os.path as path
import funclite.iolib as iolib
import os

# ROOT = 'S:/Quality Assurance/Laboratories/Graham/CSW21_NetZero_Chemical cores'

ROOT = 'S:/Quality Assurance/Laboratories/Graham/CS21 Chemical cores'
RENAME_IN_PLACE = False


def get_name_netzero(fname):
    if len(fname) > 21: return fname
    n = fname.replace('NETZ_', '')
    n = n.replace('.JPG', '')
    n = n.replace('_', '')
    n = n.replace(' ', '')
    s = 'CSW21 %s.JPG' % n
    return s


def get_cs21(fname):
    n = fname.replace('UKSC_', '')
    n = n.replace('.JPG', '')
    n = n.replace(' ', '')
    s = 'CS21 %s.JPG' % n
    return s

i = sum([1 for _ in iolib.file_list_generator_dfe(ROOT, '*.JPG', recurse=False)])
PP = iolib.PrintProgress(i)

for d, f, e, ffn in iolib.file_list_generator_dfe(ROOT, '*.JPG', recurse=False):
    # directory basename
    renamed_fld = path.normpath('%s/renamed' % d)
    iolib.create_folder(renamed_fld)

    # bn = '%s%s' % (path.basename(d), e)

    if 'NetZero' in ROOT:
        new_file_name = get_name_netzero(f)
    elif 'CS21' in ROOT:
        new_file_name = get_cs21(f)
    else:
        raise ValueError

    fnew = path.normpath('%s/%s' % (renamed_fld, new_file_name))

    if RENAME_IN_PLACE:
        os.rename(ffn, fnew)
    else:
        if iolib.file_exists(fnew):
            raise ValueError('%s exists' % fnew)
        iolib.file_copy(ffn, fnew)
    PP.increment()



print('Done')