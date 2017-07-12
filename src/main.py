#
# Downloads, processes data and trains a neural network to identify handwritten digits from MNIST
#
# Nicholas Brawand - nicholasbrawand@gmail.com
#
import tools
import os
import test

dataUrlList = [
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
]

# download data
print("Downloading data")
outFils = []
for l in dataUrlList:
    if not os.path.isfile(os.path.basename(l)):
        outFils.append(tools.DownloadData(l))
    else:
        print(os.path.basename(l), 'already exists')
        outFils.append(tools.DownloadData(l))

# unzip data
imgFils = []
for l in outFils:
    newFil = l.replace('.gz', '')

    if not os.path.isfile(newFil):
        tools.UnzipFil(l, newFil)
    else:
        print(newFil, 'already exists')

    imgFils.append(newFil)

# imgFils are in the following order:
#
# imgFils = 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
#           't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'

y_train = tools.load_mnist_labels(imgFils[1])
X_train = tools.load_mnist_images(imgFils[0], len(y_train))

# plot first few images to test import
test.CreateSamplePlot(X_train, y_train)
test.PlotSingleKind(X_train, y_train, 25, 4)
test.PlotSingleKind(X_train, y_train, 25, 7)
