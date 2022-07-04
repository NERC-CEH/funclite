import os.path as path
import funclite.iolib as iolib
import funclite.xmllib as xmllib
from funclite.stringslib import get_between

VS_SNIPPET_ROOT = 'C:/development/python/snippets'
JETBRAINS_SNIPPET_FILE = 'C:/Users/gramon/AppData/Roaming/JetBrains/PyCharmCE2021.2/settingsRepository/repository/templates/User Python.xml'


for fld, fname, _, fqn in iolib.file_list_generator_dfe(VS_SNIPPET_ROOT, '*.snippet', recurse=True):
    with open(fqn, 'r') as F:
        txt = F.read()
    code = get_between(txt, 'CDATA[', ']]></Code>')
    desc = get_between(txt,  '<Description>', '</Description>')
    name = 'T_' % fname  # make the name from the file name

    X = xmllib.XML(JETBRAINS_SNIPPET_FILE)

    # TODO Open the jetbrains file and write this code into it!JET




