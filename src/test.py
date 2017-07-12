#
# Contains testing tools
#
# Nicholas Brawand - nicholasbrawand@gmail.com
#
import tools
import math


def CreateSamplePlot(X_train, y_train):
    """Plot the first 10 samples in X_train with unique y_train"""
    outFil = '../figures/mnist_all.png'
    #print('testing image import see: ', outFil)
    imgs = []
    for i in range(10):
        imgs.append(X_train[y_train == i][0].reshape(28, 28))
    return tools.PlotGrid(2, 5, imgs, outFil)

def PlotSingleKind(X_train, y_train, num, kind):
    """plot first num samples in X_train with y_train==kind"""
    outFil = '../figures/mnist_'+str(kind)+'.png'
    #print('testing image import see: ', outFil)
    imgs = []
    tmp = X_train[y_train == kind]
    for i in range(num):
        imgs.append(tmp[i].reshape(28, 28))
    tools.PlotGrid(int(math.sqrt(num))+1, int(math.sqrt(num)), imgs, outFil)
    return
