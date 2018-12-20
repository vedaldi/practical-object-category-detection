import lab
import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import cProfile
import pstats
import PIL

np.set_printoptions(precision=4)

im_pil = PIL.Image.open('data/pos/52163-large.jpg').resize([64, 64], PIL.Image.LANCZOS)
im_pil.save('/tmp/test.png')
x = lab.pil_to_torch(im_pil)
x = x.mean(1, keepdim=True)
net = lab.HOGNet()
hog = net(x)
np.savetxt('/tmp/x8.txt', hog.detach().numpy().reshape(-1))


plt.figure(4)
lab.imarraysc(lab.t2im(hog), spacing=1)

plt.figure(5)
hogim = net.to_image(hog)
lab.imarraysc(hogim)

plt.figure(6)
hog_=lab.flip_hog(hog)
hogim = net.to_image(hog_)
lab.imarraysc(hogim)

plt.pause(0)
