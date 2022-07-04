# geetree.py
import funclite.iolib as _iolib
import xml.etree.ElementTree as _ET
from collections import defaultdict


class XML:
    """Load XML from a string or file.

    asdict:
        get the XML as a dict
    doc:
        the ElementTree parse tree object getroot

    See https://docs.python.org/3/library/xml.etree.elementtree.html for
    operations on the doc root etc.

    Example:
    XML('<c><a>test</a><b>test1</b></c>').asdict()
    {'c': {'b': 'test1', 'a': 'test'}}
    """

    def __init__(self, file_or_str):
        if _iolib.file_exists(file_or_str):
            self.doc = _ET.parse(file_or_str)
        else:
            self.doc = _ET.fromstring(file_or_str)

        if not isinstance(self.doc, _ET.Element):
            self.doc = self.doc.getroot()

    def asdict(self):
        assert self.doc, 'No valid XML document found'
        return XML._asdict(self.doc)


    @staticmethod
    def _asdict(t):
        d = {t.tag: {} if t.attrib else None}
        children = list(t)
        if children:
            dd = defaultdict(list)
            for dc in map(XML._asdict, children):
                for k, v in dc.items():
                    dd[k].append(v)
            d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
        if t.attrib:
            d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[t.tag]['#text'] = text
            else:
                d[t.tag] = text
        return d





if __name__ == "__main__":
    s = '<c><a>test</a><b>test1</b></c>'
    G = XML(s)
    D = G.asdict()
    print(D)
