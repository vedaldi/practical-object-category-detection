import lab
import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import cProfile
import pstats

np.set_printoptions(precision=4)

if False:
    boxes1=torch.Tensor([
        [10, 20, 30, 40],
        [11, 21, 31, 41],
        [20, 30, 40, 50],
        [30, 40, 50, 60]
    ])
    boxes2=torch.Tensor([
        [13, 23, 33, 43],
        [23, 33, 43, 53]
    ])

    overlaps = lab.box_overlap(boxes1,boxes2)
    print(overlaps)

    plt.figure(1)
    results = lab.eval_detections(boxes2, torch.tensor([False,False]), boxes1)

    plt.gca().relim()
    plt.gca().autoscale_view()
    print(results)
    plt.pause(0)

x = lab.imread('data/mandatory.jpg')
#x = lab.imread('data/signs/00030.jpeg')
x = x.mean(1, keepdim=True)
net = lab.HOGNet()

#cProfile.run('hog = net(x)', sort='line')
torch.set_grad_enabled(False)
for t in range(10):
    hog = net(x)
np.savetxt('/tmp/x8.txt', hog.detach().numpy().reshape(-1))

plt.figure(4)
lab.imarraysc(lab.t2im(hog), spacing=1)

plt.figure(5)
hogim = net.to_image(hog)
lab.imarraysc(hogim)
plt.pause(0)
pass