import lab
import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

import numpy as np
np.set_printoptions(precision=4)

x = lab.imread('data/signs-sample-image.jpg')
#x = x.mean(1, keepdim=True)

#x = x[:,:,:64,:64]
net = lab.HOGNet()
hog = net(x)

np.savetxt('/tmp/x8.txt', hog.detach().numpy().reshape(-1))


hogim = net.to_image(hog)


plt.figure(4)
lab.imarraysc(lab.t2im(hog), spacing=1)

plt.figure(5)
lab.imarraysc(hogim)

plt.pause(0)
pass