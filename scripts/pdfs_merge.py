"""
Merge pdfs in a folder into a single pdf.'

Examples:
    python.exe pdfs_merge.py "C:\temp" "C:\temp\my.pdf" -r
"""

import argparse
from os.path import normpath
import os.path as path
import os

import funclite.iolib as iolib
import docs.topdf as topdf



def main():
    """main"""
    cmdline = argparse.ArgumentParser(description=__doc__)  # use the module __doc__
    cmdline.add_argument('root', help='Root folder to look for your images.')
    cmdline.add_argument('saveto', help='pdf file name')
    cmdline.add_argument('-r', '--recurse', help='Recurse <root>', action='store_true')
    cmdline.add_argument('-o', '--overwrite', help='Recurse <root>', action='store_true')
    args = cmdline.parse_args()

    outname = path.normpath(args.saveto)
    if args.overwrite and iolib.file_exists(outname):
        raise FileExistsError('File %s exists. Use option -o to overwrite.')

    root = normpath(args.root)
    chunk = []


    print('Building file lists ....')
    for _, _, _, fqn in iolib.file_list_generator_dfe(root, '*.pdf', args.recurse):
        chunk.append(fqn)
    chunk.sort()

    print('Merging pdfs .. this may take some time ....')
    topdf.merge_pdf_by_list(chunk, outname)
    print('Done. Create pdf %s.' % outname)


# TODO Complete this routine to chunk merge pdfs
if __name__ == "__main__":
    main()
