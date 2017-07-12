#
# Downloads, processes data and trains a neural network to identify handwritten digits from MNIST
#
# Nicholas Brawand - nicholasbrawand@gmail.com
#
import tools
import os

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

# unzip data
for l in outFils:
    newFil = l.replace('.gz','')
    if not os.path.isfile(newFil):
        tools.UnzipFil(l, newFil)
    else:
        print(newFil, 'already exists')

X_train, y_train = tools.load_mnist('', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
