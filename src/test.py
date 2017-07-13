#
# Contains testing tools
#
# Nicholas Brawand - nicholasbrawand@gmail.com
#
from . import tools
import math


def CreateSamplePlot(X_train, y_train, outFil):
    """Plot the first 10 samples in X_train with unique y_train"""
    imgs = []
    for i in range(10):
        imgs.append(X_train[y_train == i][0].reshape(28, 28))
    return tools.PlotGrid(2, 5, imgs, outFil)

def PlotSingleKind(X_train, y_train, num, kind, outFil):
    """plot first num samples in X_train with y_train==kind"""
    imgs = []
    tmp = X_train[y_train == kind]
    for i in range(num):
        imgs.append(tmp[i].reshape(28, 28))
    tools.PlotGrid(int(math.sqrt(num))+1, int(math.sqrt(num)), imgs, outFil)
    return
