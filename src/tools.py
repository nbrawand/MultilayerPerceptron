#
# Contains misc tools such as downloading data
#
# Nicholas Brawand - nicholasbrawand@gmail.com
#
import os
import struct
import numpy as np
import urllib.request as urllib2
import gzip
import matplotlib.pyplot as plt


def DownloadData(link):
    """Downloads file from link and returns the file name."""
    # import urllib2

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
    inF = gzip.open(inFil, 'rb')
    outF = open(outFil, 'wb')
    outF.write(inF.read())
    inF.close()
    outF.close()
    return

def load_mnist_images(path, lda):
    """
    Load MNIST image data from path and return the images
    lda is the leading dimension for reshape of the image.
    normally lda is the len of the labels
    """
    images_path = os.path.join(path)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(lda, 784)

    return images


def load_mnist_labels(path):
    """Load MNIST label data from path and return the labels"""
    labels_path = os.path.join(path)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    return labels


def PlotGrid(nrows, ncols, imgs, outFil):
    """save a grid of images to outFil."""

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,)
    ax = ax.flatten()

    for i in range(len(imgs)):
        img = imgs[i]
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        fig = plt.savefig(outFil, dpi=300)
    return fig
