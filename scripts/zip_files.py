"""
Zip files in a directory, matching names

Examples:
    > python.exe zip_files.py C:\temp\files C:\temp\file.zip -o -name_match 123,345,678
    . Birds aerial_a3
    > python.exe zip_files.py
        "S:/SPECIAL-ACL/ERAMMP2 Survey Restricted/2022/survey/square packs/maps/common/aerial_a3_5k"
        "S:\SPECIAL-ACL\ERAMMP2 Survey Restricted\2022\survey\square packs\birds\collated\aerial_a3.zip"
        -o
        -n 2665,3522,4967,5030,5206,5774,6648,8459,9808,10383,10576,11237,11310,12305,12729,13170,13188,13913,13917,14009,14821,15863,16065,16100,16256,16305,16714,17153,18189,18859,19058,19434,20081,20943,22020,23059,23254,23852,24270,24296,25533,27835,30194,30589,30816,30818,31037,31230,31437,31877,32469,33098,33714,33879,33894,34304,34727,34790,35185,35573,36000,36407,36533,36665,37780,38115,38333,38967,39617,40071,40445,41938,42167,43153,43396,43434,43579,44196,45655,45865
"""
import argparse

import docs.zip as doczip





def main():
    """main"""
    cmdline = argparse.ArgumentParser(description=__doc__)

    cmdline.add_argument('root', help='Root folder to find files to zip e.g. "C:/temp"')
    cmdline.add_argument('dest', help='Zip file to save e.g. "C:/temp/my.zip"')


    f = lambda s: [str(item) for item in s.split(',')]
    cmdline.add_argument('-n', '--name_match', type=f, help='Comma delimited list of strings to match to file names, eg -f fileA,fileB')
    cmdline.add_argument('-o', '--overwrite', help='Allow overwrite', action='store_true')
    args = cmdline.parse_args()
    name_match = tuple(args.name_match) if args.name_match else ()
    doczip.zipdir(args.root, args.dest, name_match)
    print('Done. Zip %s created' % args.dest)


if __name__ == "__main__":
    main()
