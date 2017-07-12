#
# Contains misc tools such as downloading data
#
# Nicholas Brawand - nicholasbrawand@gmail.com
#


def DownloadData(link):
    """Downloads file from link and returns the file name."""
    # import urllib2
    import urllib.request as urllib2
    import os

    outFil = os.path.basename(link)

    try:
        # open link
        f = urllib2.urlopen(link)

        # write file
        with open(outFil, "wb") as local_file:
            local_file.write(f.read())

        # return file name
        return outFil

    # handle errors
    except:
        print('Error in DownloadData:', link)
        pass


def UnzipFil(inFil, outFil):
    """uncompress inFil and write to outFil"""
    import gzip
    inF = gzip.open(inFil, 'rb')
    outF = open(outFil, 'wb')
    outF.write(inF.read())
    inF.close()
    outF.close()
    return

import os
import struct
import numpy as np


def load_mnist(path, kind='train'):
    """LoadMNISTdatafrom`path`"""
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
        lbpath.read(8))
        labels = np.fromfile(lbpath,
        dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
        imgpath.read(16))
        images = np.fromfile(imgpath,
        dtype=np.uint8).reshape(len(labels), 784)

    return images,labels
