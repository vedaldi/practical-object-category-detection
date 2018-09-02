import lab
import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

import numpy as np
np.set_printoptions(precision=4)

x = lab.imread('data/signs-sample-image.jpg')
x = x.mean(1, keepdim=True)
net = lab.HOGNet()
hog = net(x)
        
plt.pause(0)

        