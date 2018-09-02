import math
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional  as F
from matplotlib import pyplot as plt
from PIL import Image

def t2im(x):
    """Rearrange the N x K x H x W to have shape (NK) x 1 x H x W.

    Arguments:
        x {torch.Tensor} -- A N x K x H x W tensor.

    Returns:
        torch.Tensor -- A (NK) x 1 x H x W tensor.
    """
    return x.reshape(-1, *x.shape[2:])[:,None,:,:]

def pil_to_torch(x):
    x = np.array(x)
    if len(x.shape) == 2:
        x = x[:,:,None]
    return torch.tensor(x, dtype=torch.float32).permute(2,0,1)[None,:]/255

def imread(file):
    """Read the image `file` as a PyTorch tensor.

    Arguments:
        file {str} -- The path to the image.

    Returns:
        torch.Tensor -- The image read as a 3 x H x W tensor in the [0, 1] range.
    """
    # Read an example image as a NumPy array
    return pil_to_torch(Image.open(file))

def imsc(im, *args, quiet=False, **kwargs):
    """Rescale and plot an image represented as a PyTorch tensor.

     The function scales the input tensor im to the [0 ,1] range.

    Arguments:
        im {torch.Tensor} -- A 3 x H x W or 1 x H x W tensor.

    Keyword Arguments:
        quiet {bool} -- Do not plot. (default: {False})

    Returns:
        torch.Tensor -- The rescaled image tensor.
    """
    handle = None
    with torch.no_grad():
        im = im - im.min() # make a copy
        im.mul_(1/im.max())
        if not quiet:
            bitmap = im.expand(3, *im.shape[1:]).permute(1,2,0).numpy()
            handle = plt.imshow(bitmap, *args, **kwargs)
            ax = plt.gca()
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    return im, handle

def imarraysc(tiles, spacing=0, quiet=False):
    """Plot the PyTorch tensor `tiles` with dimesion N x C x H x W as a C x (MH) x (NW) mosaic.

    The range of each image is individually scaled to the range [0, 1].

    Arguments:
        tiles {[type]} -- [description]

    Keyword Arguments:
        spacing {int} -- Thickness of the border (infilled with zeros) around each tile (default: {0})
        quiet {bool} -- Do not plot the mosaic. (default: {False})

    Returns:
        torch.Tensor -- The mosaic as a PyTorch tensor.
    """
    handle = None
    num = tiles.shape[0]
    num_cols = math.ceil(math.sqrt(num))
    num_rows = (num + num_cols - 1) // num_cols
    c = tiles.shape[1]
    h = tiles.shape[2]
    w = tiles.shape[3]
    mosaic = torch.zeros(c,
      h*num_rows + spacing*(num_rows-1),
      w*num_cols + spacing*(num_cols-1))
    for t in range(num):
        u = t % num_cols
        v = t // num_cols
        tile = tiles[t]
        mosaic[0:c,
          v*(h+spacing) : v*(h+spacing)+h,
          u*(w+spacing) : u*(w+spacing)+w] = imsc(tiles[t], quiet=True)[0]
    return imsc(mosaic, quiet=quiet)

class HOGNet(nn.ModuleDict):
    def __init__(self):
        super(nn.ModuleDict, self).__init__()
        with torch.no_grad():
            self.use_bilinear_orientation_pooling = False
            self.num_orientations = 9
            self.bin_size = 8

            # Spatial derivative filters along num_orientations directions
            du = torch.tensor([
                [ 0, 0, 0],
                [-1, 0, 1],
                [ 0, 0, 0]
            ], dtype=torch.float32) / 2
            dv = du.t()
            spatial_ders = []
            for i in range(2 * self.num_orientations):
                t = (2 * math.pi) / (2 * self.num_orientations) * i
                der = math.cos(t) * du + math.sin(t) * dv
                spatial_ders.append(der[None,None,:])
            spatial_ders = torch.cat(spatial_ders, 0)

            self['spatial_ders'] = nn.Conv2d(1, self.num_orientations, 3,
                stride=1, padding=1, bias=False)
            self['spatial_ders'].weight.data = spatial_ders
            
            if False:
                plt.figure(1)
                lab.imarraysc(spatial_ders, spacing=1)
                plt.pause(0)

            # Bilinear spatial binning into bin_size x bin_size cells
            a = 1 - torch.abs((torch.Tensor(range(1,2*self.bin_size+1)) - (2*self.bin_size+1)/2)/self.bin_size)
            bilinear_filter = a.reshape(-1,1) @ a.reshape(1,-1)
            bilinear_filter = bilinear_filter[None,None,:]
            bilinear_filter = bilinear_filter.expand(2 * self.num_orientations,*bilinear_filter.shape[1:])

            self['spatial_pool'] = nn.Conv2d(2 * self.num_orientations, 2 * self.num_orientations, 2 * self.bin_size,
            stride=self.bin_size, padding=self.bin_size//2, bias=False, groups=2 * self.num_orientations)
            self['spatial_pool'].weight.data = bilinear_filter

            # 2x2 block pooling
            self['block_pool'] = nn.Conv2d(self.num_orientations, 1, 2, bias=False)
            self['block_pool'].weight.data = torch.ones(1,self.num_orientations,2,2)

            # (1,1,1,1) replication padding
            self['padding'] = nn.ReplicationPad2d(1)

    def angular_binning(self, oriented_gradients):
        # Compute the norm of the gradient from the directional derivatives.
        # This can be done by summing their squares.
        n2 = torch.sum(oriented_gradients * oriented_gradients, 1) / self.num_orientations
        n = torch.sqrt(n2)

        # Get the cosine of the angle between the gradients and the
        # reference directions.
        cosines = oriented_gradients / torch.clamp(n, min=1e-15)
        cosines = torch.clamp(cosines, -1, 1)

        # Recover the angles from the cosines.
        angles = torch.acos(cosines)

        # Get the bilinear angular binning coefficients.
        bin_weights = 1 - (2 * self.num_orientations)/(2 * math.pi) * angles
        bins = torch.clamp(bin_weights, min=0)

        return n * bins

    def block_normalization(self, cells):
        # Compute the unoriented gradient cells
        ucells = cells[:,:self.num_orientations,:,:] + cells[:,self.num_orientations:,:,:]

        # Comptue the norm of 2 x 2 blocks of unoriented gradient cells
        squares = ucells * ucells
        sums = self['block_pool'](squares)
        norms = torch.sqrt(torch.clamp(sums, min=1e-6))

        # Put unoriented and oriented gradients together
        cells = torch.cat((cells,ucells),1)

        # Normalize and clmap each cell as if part of each 2x2 block 
        # individually, then average the results.
        factors = 1 / norms
        factors = self['padding'](factors)
        factors = factors.expand((-1, 3 * self.num_orientations, -1, -1))
        
        ncells = (
            torch.clamp(cells * factors[:, :, :-1, :-1], max=0.2) +
            torch.clamp(cells * factors[:, :, :-1, +1:], max=0.2) +
            torch.clamp(cells * factors[:, :, +1:, +1:], max=0.2) +
            torch.clamp(cells * factors[:, :, +1:, :-1], max=0.2)
        ) * 0.5

        return ncells

    def forward(self, im):
        oriented_gradients = self['spatial_ders'](im)
        oriented_histograms = self.angular_binning(oriented_gradients)    
        cells = self['spatial_pool'](oriented_histograms)
        ncells = self.block_normalization(cells)
        hog = torch.clamp(ncells, max=0.2)

        if False:
            np.savetxt('/tmp/x8.txt', ncells.detach().numpy().reshape(-1))
            plt.figure(1)
            lab.imarraysc(lab.t2im(oriented_gradients), spacing=1)
            plt.figure(2)
            lab.imarraysc(lab.t2im(oriented_histograms), spacing=1)
            plt.figure(3)
            lab.imarraysc(lab.t2im(cells), spacing=1)
            plt.figure(4)
            lab.imarraysc(lab.t2im(ncells), spacing=1)
            plt.pause(1e-4)
