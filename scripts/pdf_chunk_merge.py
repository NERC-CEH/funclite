# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument
__doc__ = ('Merge pdfs in a folder, specifying number chunks.\n'
           'syntax: pdfmerge.py <root> <n> [-r]\n'
           'root: root folder to look for images\n'
           'saveto: file to save to\n'
           'n: number of pdfs to chunk together\n'
           '-r: if present, recurse through folders in root\n'
           '-o: force overwrite of the output pdf\n\n'
           'Example:\n'
           'img2pdf "C:\temp" -r')

import argparse
from os.path import normpath
import copy

import funclite.iolib as iolib
import docs.topdf as topdf


def _make_name(chunk, suffix=''):
    """(list) -> str
    Make filename
    """
    s = '_'.join(iolib.get_file_parts(chunk)[1])
    if suffix: s = '%s_%s' % (s, suffix)
    return s


def main():
    """main"""
    cmdline = argparse.ArgumentParser(description=__doc__)  # use the module __doc__
    cmdline.add_argument('root', help='Root folder to look for your images.')
    cmdline.add_argument('saveto', help='Root folder to save the created pdf.')
    cmdline.add_argument('n', help='Number of files to chunk together')
    cmdline.add_argument('-r', '--recurse', help='Recurse <root>', action='store_true')
    cmdline.add_argument('-o', '--overwrite', help='Allow overwriting of existing pdfs', action='store_true')
    args = cmdline.parse_args()
    saveto = normpath(args.saveto)
    root = normpath(args.root)
    n = int(args.n)
    chunks = []
    chunk = []

    # if n is 2, chunks gonna look like this:
    # [['1.pdf','2.pdf'], ['3.pdf','4.pdf', ....]

    print('Building file lists ....')
    for _, _, _, fqn in iolib.file_list_generator_dfe(root, '*.pdf', args.recurse):
        if len(chunk) == n:
            c = copy.copy(chunk)
            chunks.append(c)
            chunk = []
        chunk.append(fqn)

    print('Merging pdfs ....')
    PP = iolib.PrintProgress(len(chunks))
    for chunk in chunks:
        i = 0
        while True:
            # this loop incase we have a situation with duplicate filenames, unlikely but could happen
            suffix = '' if i == 0 else str(i)
            outname = '%s/%s.pdf' % (saveto, _make_name(chunk, suffix))
            outname = normpath(outname)
            if not iolib.file_exists(outname): break
            i += 1

        topdf.merge_pdf_by_list(chunk, outname)
        PP.increment()


# TODO Complete this routine to chunk merge pdfs
if __name__ == "__main__":
    raise NotImplementedError
    main()
