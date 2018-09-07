import math
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional  as F
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

def reset_random_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    def __init__(self, cell_size=8, num_orientations=9):
        super(nn.ModuleDict, self).__init__()
        with torch.no_grad():
            self.num_orientations = num_orientations
            self.cell_size = cell_size

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

            # Same as above, but for color images
            self['spatial_ders_rgb'] = nn.Conv2d(3, self.num_orientations*3, 3,
                stride=1, padding=1, bias=False, groups=3)
            self['spatial_ders_rgb'].weight.data = spatial_ders.repeat(3,1,1,1)
            
            if False:
                plt.figure(1)
                lab.imarraysc(spatial_ders, spacing=1)
                plt.pause(0)

            # Bilinear spatial binning into cell_size x cell_size cells
            a = 1 - torch.abs((torch.Tensor(range(1,2*self.cell_size+1)) - (2*self.cell_size+1)/2)/self.cell_size)
            bilinear_filter = a.reshape(-1,1) @ a.reshape(1,-1)
            bilinear_filter = bilinear_filter[None,None,:]
            bilinear_filter = bilinear_filter.expand(2 * self.num_orientations,*bilinear_filter.shape[1:])

            self['spatial_pool'] = nn.Conv2d(2 * self.num_orientations, 2 * self.num_orientations, 2 * self.cell_size,
            stride=self.cell_size, padding=self.cell_size//2, bias=False, groups=2 * self.num_orientations)
            self['spatial_pool'].weight.data = bilinear_filter

            # 2x2 block pooling
            self['block_pool'] = nn.Conv2d(self.num_orientations, 1, 2, bias=False)
            self['block_pool'].weight.data = torch.ones(1,self.num_orientations,2,2)

            # (1,1,1,1) replication padding
            self['padding'] = nn.ReplicationPad2d(1)

            # Glyphs for rendering the descriptor
            gs = 21
            u = torch.linspace(-1,1,gs).reshape(-1,1).expand(-1,gs)
            v = u.t()
            n = torch.sqrt(u*u + v*v) + 1e-12
            glyphs = []
            for i in range(self.num_orientations):
                t = (math.pi / self.num_orientations) * i
                cos = (math.sin(t) * u + math.cos(-t) * v) / n 
                glyph = torch.exp(-torch.abs(cos) * 15)
                dim = 1 if t < math.pi/4 or t > 3*math.pi/4 else 0
                glyph = (glyph == glyph.max(dim=dim, keepdim=True)[0]).expand_as(glyph)
                glyph *= (n < 1)
                glyph = glyph.to(torch.float32)
                glyphs.append(glyph[None,None,:,:])
            glyphs = torch.cat(glyphs, 0)

            self['render'] = nn.ConvTranspose2d(self.num_orientations, 1, gs,
                stride=gs, bias=False)#, output_padding=gs-1)
            self['render'].weight.data = glyphs

            if False:
                plt.figure(100)
                imarraysc(t2im(glyphs))
                plt.pause(0)            

    def reduce_oriented_gradients(self, oriented_gradients):
        no = self.num_orientations
        nc = oriented_gradients.shape[1] // (2*no)
        og = [None,None,None]
        n = [None,None,None]
        for c in range(nc):
            og[c] = oriented_gradients[:,2*no*c:2*no*(c+1),:,:]
            n2 = torch.sum(og[c] * og[c], 1, keepdim=True) / no
            n[c] = torch.sqrt(n2)
        n_max = n[0]
        for c in range(1,nc):
            n_max = torch.max(n[c], n_max)
        return (
            og[0] * (n_max == n[0]).to(torch.float32) +
            og[1] * (n_max == n[1]).to(torch.float32) +
            og[2] * (n_max == n[2]).to(torch.float32)
        )

    def angular_binning(self, oriented_gradients):
        # Compute the norm of the gradient from the directional derivatives.
        # This can be done by summing their squares.
        n2 = torch.sum(oriented_gradients * oriented_gradients, 1, keepdim=True) / self.num_orientations
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
        #factors = factors.expand((-1, 3 * self.num_orientations, -1, -1))
        
        ncells = (
            torch.clamp(cells * factors[:, :, :-1, :-1], max=0.2) +
            torch.clamp(cells * factors[:, :, :-1, +1:], max=0.2) +
            torch.clamp(cells * factors[:, :, +1:, +1:], max=0.2) +
            torch.clamp(cells * factors[:, :, +1:, :-1], max=0.2)
        ) * 0.5

        return ncells

    def forward(self, im):
        if im.shape[1] == 1:
            oriented_gradients = self['spatial_ders'](im)
        else:
            oriented_gradients = self['spatial_ders_rgb'](im)
            oriented_gradients = self.reduce_oriented_gradients(oriented_gradients)
        oriented_histograms = self.angular_binning(oriented_gradients)    
        cells = self['spatial_pool'](oriented_histograms)
        hog = self.block_normalization(cells)
        return hog

    def to_image(self, hog):
        with torch.no_grad():
            weight = (
                hog[:,:self.num_orientations,:,:] + 
                hog[:,self.num_orientations:self.num_orientations*2,:,:] +
                hog[:,self.num_orientations*2:,:,:]
            )
            im = self['render'](weight)
            im = torch.clamp(im, weight.min(), weight.max())
        return im

def load_data(meta_class='all'):
    imdb = torch.load('data/signs-data.pth')
    if meta_class is 'prohibitory':
        meta_labels = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
    elif meta_class is 'mandatory':
        meta_labels = [33, 34, 35, 36, 37, 38, 39, 40]
    elif meta_class is 'danger':
        meta_labels = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    elif meta_class is 'all':
        meta_labels = range(43)
    else:
        raise ValueError('The value of meta_label is not recognized.')

    for subset in ['train', 'val']:
        labels = [x.item() for x in imdb[subset]['box_labels']]
        sel = [i for i, x in enumerate(labels) if x in meta_labels]
        imdb[subset]['boxes'] = imdb[subset]['boxes'][sel,:]
        imdb[subset]['box_images'] = [imdb[subset]['box_images'][i] for i in sel]
        imdb[subset]['box_labels'] = imdb[subset]['box_labels'][sel]
        imdb[subset]['box_patches'] = imdb[subset]['box_patches'][sel,:,:,:]
        imdb[subset]['images'] = sorted(list(set(imdb[subset]['box_images'])))

    return imdb

def box_overlap(boxes1, boxes2, measure='iou'):
    """Compute the intersection over union of bounding boxes

    Arguments:
        boxes1 {torch.Tensor} -- N1 x 4 tensor with [x0,y0,x1,y1] for N boxes.
                                 For one box, a 4 tensor is also supported.
        boxes2 {torch.Tensor} -- N2 x 4 tensor.

    Returns:
        torch.Tensor -- N1 x N2 tensor with the IoU overlaps.
    """
    boxes1 = boxes1.reshape(-1,1,4)
    boxes2 = boxes2.reshape(1,-1,4)
    areas1 = torch.prod(boxes1[:,:,:2] - boxes1[:,:,2:], 2)
    areas2 = torch.prod(boxes2[:,:,:2] - boxes2[:,:,2:], 2)

    max_ = torch.max(boxes1[:,:,:2], boxes2[:,:,:2])
    min_ = torch.min(boxes1[:,:,2:], boxes2[:,:,2:])
    intersections = torch.prod(torch.clamp(min_ - max_, min=0), 2)

    overlaps = intersections / (areas1 + areas2 - intersections)
    return overlaps

def plot_box(box, color='y'):
    r1 = matplotlib.patches.Rectangle(box[:2],
                                     box[2]-box[0], box[3]-box[1],
                                     facecolor='none', linestyle='solid',
                                     edgecolor=color, linewidth=3)
    r2 = matplotlib.patches.Rectangle(box[:2],
                                     box[2]-box[0], box[3]-box[1],
                                     facecolor='none', linestyle='solid',
                                     edgecolor='k', linewidth=5)
    plt.gca().add_patch(r2)
    plt.gca().add_patch(r1)
    
def pr(labels, scores, misses=0):
    "Plot the precision-recall curve."
    scores, perm = torch.sort(scores, descending=True)
    labels = labels[perm]
    tp = (labels > 0).to(torch.float32)
    ttp = torch.cumsum(tp, 0)
    precision = ttp / torch.arange(1, len(tp)+1, dtype=torch.float32)
    recall = ttp / torch.clamp(tp.sum() + misses, min=1)
    # Labels may contain no positive labels (perhaps because misses>0)
    # which would case mean() to nan
    ap = precision[tp > 0]
    ap = ap.mean() if len(ap) > 0 else 0
    plt.plot(recall.numpy(), precision.numpy())
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0,1.01)
    plt.ylim(0,1.01)
    return precision, recall, ap

def eval_detections(gt_boxes, boxes, threshold=0.5, plot=False, gt_difficult=None):
    with torch.no_grad():
        # Compute the overlap between ground-truth boxes and detected ones
        overlaps = box_overlap(boxes, gt_boxes)

        # Match each box to a gt box
        overlaps, box_to_gt = torch.max(overlaps, dim=1)
        matched = overlaps > threshold
        labels = -torch.ones(len(boxes))
        labels[matched] = +1

        # Discount the boxes that match difficult gts
        if gt_difficult is not None:
            discounted = matched & gt_difficult[box_to_gt]
            matched &= discounted ^ 1 # logic negation as XOR
            labels[discounted.nonzero()] = 0

        misses = 0
        gt_to_box = torch.full((len(gt_boxes),), -1, dtype=torch.int64)
        for i in range(len(gt_boxes)):
            if gt_difficult is not None and gt_difficult[i]:
                continue
            j = torch.nonzero((box_to_gt == i) & matched)
            if len(j) == 0:
                misses += 1
            else:
                gt_to_box[i] = j[0]
                labels[j[1:]] = -1
            matched[j] = 0
        
        if plot:
            for box in gt_boxes:
                plot_box(box, color='y')
            for box, label in zip(boxes, labels):
                plot_box(box, color='g' if label > 0 else 'r')

        return {
            'gt_to_box': gt_to_box,
            'box_to_gt': box_to_gt,
            'labels': labels,
            'misses': misses,
        }

def svm_sdca(x, c, lam=0.01, epsilon=0.0005, num_epochs=1000):
    "Train an SVM using the SDCA algorithm."
    with torch.no_grad():
        xb = 1
        n = x.shape[0]
        d = x.shape[1]
        alpha = torch.zeros(n)
        w = torch.zeros(d)
        b = 0
        A = ((x * x).sum(1) + (xb * xb)) / (lam * n)

        lb_log = []
        ub_log = []
        lb = 0

        for epoch in range(num_epochs):
            perm = np.random.permutation(n)
            for i in perm:
                B = x[i] @ w + xb * b
                dalpha = (c[i] - B) / A[i]
                dalpha = c[i] * max(0, min(1, c[i] * (dalpha + alpha[i]))) - alpha[i]
                lb -= (A[i]/2 * (dalpha**2) + (B - c[i])*dalpha) / n
                w += (dalpha / (lam * n)) * x[i]
                b += (dalpha / (lam * n)) * xb
                alpha[i] += dalpha  

            scores = x @ w + xb * b
            ub = torch.clamp(1 - c * scores, min=0).mean() + (lam/2) * (w * w).sum() + (lam/2) * (b * b)
            lb_log.append(lb.item())
            ub_log.append(ub.item())        
            finish = (epoch + 1 == num_epochs) or (ub_log[-1] - lb_log[-1] < epsilon)

            if (epoch % 10 == 0) or finish:
                print(f"SDCA epoch: {epoch: 2d} lower bound: {lb_log[-1]:.3f} upper bound: {ub_log[-1]:.3f}")

            if ((epoch > 0) and (epoch % 200 == 0)) or finish:
                plt.figure(1)
                plt.clf()
                plt.title('SDCA optimization')
                plt.plot(lb_log)
                plt.plot(ub_log, '--')
                plt.legend(('lower bound', 'upper bound'))
                plt.xlabel('iteration')
                plt.ylabel('energy')
                plt.pause(0.0001)

            if finish:
                break

    return w, xb * b

