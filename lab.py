import math
import random
import numpy as np
import time
import copy
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
            no = self.num_orientations
            cs = self.cell_size

            # Spatial derivative filters
            d = torch.tensor([-1, 0, 1], dtype=torch.float32) / 2
            self['du']     = nn.Conv2d(1, 1, (1,3), stride=1, padding=(0,1), bias=False)
            self['dv']     = nn.Conv2d(1, 1, (3,1), stride=1, padding=(1,0), bias=False)
            self['du_rgb'] = nn.Conv2d(3, 3, (1,3), stride=1, padding=(0,1), bias=False, groups=3)
            self['dv_rgb'] = nn.Conv2d(3, 3, (3,1), stride=1, padding=(1,0), bias=False, groups=3)
            self['du'].weight.data = d.reshape(1,1,1,3).clone()
            self['dv'].weight.data = d.reshape(1,1,3,1).clone()
            self['du_rgb'].weight.data = d.reshape(1,1,1,3).expand(3,1,1,3).clone()
            self['dv_rgb'].weight.data = d.reshape(1,1,3,1).expand(3,1,3,1).clone()

            # Directional projection filters
            orient = torch.zeros((2 * no, 2, 1, 1), dtype=torch.float32)
            for i in range(2 * no):
                angle = (2 * math.pi) / (2 * no) * i
                orient[i,0] = math.cos(angle)
                orient[i,1] = math.sin(angle)

            self['orient'] = nn.Conv2d(2, 2 * no, 1, stride=1, bias=False)
            self['orient'].weight.data = orient

            # Bilinear spatial binning into cell_size x cell_size cells
            window = 1 - torch.abs((torch.Tensor(range(1, 2*cs + 1)) - (2*cs + 1)/2)/cs)
            self['poolu'] = nn.Conv2d(2*no, 2*no, (1, 2*cs), padding=(0, cs//2), stride=(1, cs), bias=False, groups=2*no)
            self['poolv'] = nn.Conv2d(2*no, 2*no, (2*cs, 1), padding=(cs//2, 0), stride=(cs, 1), bias=False, groups=2*no)
            self['poolu'].weight.data = window.reshape(1,1,1,-1).expand(2*no,-1,-1,-1).clone()
            self['poolv'].weight.data = window.reshape(1,1,-1,1).expand(2*no,-1,-1,-1).clone()

            # bilinear_filter = bilinear_filter[None,None,:]
            # bilinear_filter = bilinear_filter.expand(2 * no, *bilinear_filter.shape[1:])

            # self['spatial_pool'] = nn.Conv2d(2 * no, 2 * no, 2 * self.cell_size,
            # stride=self.cell_size, padding=self.cell_size//2, bias=False, groups=2 * no)
            # self['spatial_pool'].weight.data = bilinear_filter

            # 2x2 block pooling.
            self['block_pool'] = nn.Conv2d(self.num_orientations, 1, 2, bias=False)
            self['block_pool'].weight.data = torch.ones(1,self.num_orientations,2,2)

            # (1,1,1,1) replication padding.
            self['padding'] = nn.ReplicationPad2d(1)

            # Glyphs for rendering the descriptor.
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

    def compute_oriented_gradients(self, im):        
        if im.shape[1] == 1:
            du = self['du'](im)
            dv = self['dv'](im)
            n2 = du*du + dv*dv
        else:
            du = self['du_rgb'](im)
            dv = self['dv_rgb'](im)
            n2 = du*du + dv*dv
            n2, indices = torch.max(n2, 1, keepdim=True)
            du = torch.gather(du, 1, indices)
            dv = torch.gather(dv, 1, indices)

        # Delete boundary pixel gradients as they are subject to boundary effects.
        n2[:,:,+0,:] = 0
        n2[:,:,-1,:] = 0
        n2[:,:,:,+0] = 0
        n2[:,:,:,-1] = 0

        oriented_gradients = self['orient'](torch.cat((du,dv), 1))
        return oriented_gradients, n2

    def angular_binning(self, oriented_gradients, norms2):
        # Compute the norm of the gradient from the directional derivatives.
        # This can be done by summing their squares.
        norms = torch.sqrt(norms2)
        factors = 1 / torch.clamp(norms, min=1e-15)

        # Get the cosine of the angle between the gradients and the
        # reference directions.
        cosines = oriented_gradients * factors

        # Recover the angles from the cosines and compute the bins.
        if False:
            # This is slightly closer to the original implementation.
            cosines = torch.clamp(cosines, -1, 1)
            angles = torch.acos(cosines)
            bin_weights = 1 - (2 * self.num_orientations)/(2 * math.pi) * angles
        else:
            # Faster approximation without acos. Good to 1e-4 error w.r.t original.
            # cos x = y ~ 1 - x^2/2
            # acos y ~ sqrt(2*(x - 1))
            cosines = torch.clamp(cosines, max=1)
            angles_sqrt2 = torch.sqrt(1 - cosines)
            bin_weights = 1 - (math.sqrt(2) * self.num_orientations / math.pi) * angles_sqrt2

        # Get the bilinear angular binning coefficients.
        bins = torch.clamp(bin_weights, min=0)

        return norms * bins

    def block_normalization(self, cells):
        # Compute the unoriented gradient cells.
        ucells = cells[:,:self.num_orientations,:,:] + cells[:,self.num_orientations:,:,:]

        # Comptue the norm of 2 x 2 blocks of unoriented gradient cells.
        squares = ucells * ucells
        squares = self['padding'](squares)
        sums = self['block_pool'](squares)
        #norms = torch.sqrt(torch.clamp(sums, min=1e-6))
        norms = torch.sqrt(sums + 1e-4)

        # Put unoriented and oriented gradients together.
        cells = torch.cat((cells,ucells),1)

        # Normalize and clmap each cell as if part of each 2x2 block 
        # individually, then average the results.
        factors = 1 / norms
        #factors = self['padding'](factors)
        
        ncells = (
            torch.clamp(cells * factors[:, :, :-1, :-1], max=0.2) +
            torch.clamp(cells * factors[:, :, :-1, +1:], max=0.2) +
            torch.clamp(cells * factors[:, :, +1:, +1:], max=0.2) +
            torch.clamp(cells * factors[:, :, +1:, :-1], max=0.2)
        ) * 0.5

        return ncells

    def forward(self, im):
        oriented_gradients, norms = self.compute_oriented_gradients(im)
        oriented_histograms = self.angular_binning(oriented_gradients, norms)    
        cells = self['poolu'](oriented_histograms)
        cells = self['poolv'](cells)
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

    def detect_at_multiple_scales(self, w, scales, pil_image, use_gpu=None):
        # Decide if a GPU should be used.
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        # Wrap parameters w in a convolutional layer.
        model = nn.Conv2d(27, 1, w.shape[2:], bias=False)
        model.weight.data = w
        
        # Send model to GPU if needed.
        device = torch.device("cuda" if use_gpu else "cpu")
        hog_extractor_device = copy.deepcopy(self).to(device)
        model = model.to(device)

        # Search for strong responses across different scales.
        all_boxes = []
        all_scores = []
        all_hogs = []
        for t, scale in enumerate(scales):
            # Scale the input image.
            size = [int(round(x/scale)) for x in pil_image.size]
            scaled_image = pil_image.resize(size)

            # Skip if the scaled image is smaller than 2 x 2 HOG cells.
            if scaled_image.size[0] < 2*self.cell_size or scaled_image.size[1] < 2*self.cell_size:
                continue

            # Extract its HOG representation.
            hog = hog_extractor_device(pil_to_torch(scaled_image).to(device))

            # Skip if the HOG representation is smaller than the model.
            if hog.shape[2] < w.shape[2] or hog.shape[3] < w.shape[3]:
                continue
                
            # Apply the model convolutionally.
            scores = model(hog)
            scores = scores.to("cpu")
            
            # Get the boxes and reshape them into a N x 4 list.
            boxes = scale * boxes_for_scores(model, scores[0])
            boxes = boxes.reshape(4,-1).permute(1,0)
            scores = scores.reshape(-1)
            
            # Store for later.
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_hogs.append(hog.to("cpu"))
                                    
        # Concatenate results.
        return torch.cat(all_boxes, 0), torch.cat(all_scores, 0), all_hogs

def flip_hog(hog):
    no = 9	
    o = torch.arange(no).long()
    op = no - o
    perm = torch.cat((op, (op + no) % (2*no), (op % no) + (2*no)),0)
    return hog[:,perm,:,:].flip(3)

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

def boxes_for_scores(model, scores, cell_size=8):
    mh = model.kernel_size[0]
    mw = model.kernel_size[1]
    with torch.no_grad():
        h = scores.shape[1]
        w = scores.shape[2]
        v0 = torch.arange(h, dtype=torch.float32).reshape(1,-1,1)
        u0 = torch.arange(w, dtype=torch.float32).reshape(1,1,-1)
        v1 = v0 + mh
        u1 = u0 + mw
        boxes = torch.cat([(x * cell_size).expand(1,h,w) for x in [u0,v0,u1,v1]], 0)
    return boxes

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
    
def pr(labels, scores, misses=0, plot=True):
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
    if plot:
        plt.plot(recall.numpy(), precision.numpy())
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0,1.01)
        plt.ylim(0,1.01)
    return precision, recall, ap

    
def eval_detections(gt_boxes, boxes, threshold=0.5, plot=False, gt_difficult=None):
    if len(gt_boxes) == 0:
        return {
            'gt_to_box': [],
            'box_to_gt': torch.tensor([-1]*len(boxes)),
            'labels': torch.tensor([-1]*len(boxes)),
            'misses': 0,
        }

    with torch.no_grad():

        # Compute the overlap between ground-truth boxes and detected ones.
        overlaps = box_overlap(boxes, gt_boxes)

        # Match each box to a gt box.
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
            for box, label in reversed(list(zip(boxes, labels))):
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

def nms(boxes, scores):
    "Return a tensor of boolean values with True for the boxes to retain"
    n = len(boxes)
    scores_ = scores.clone()
    retain = torch.zeros(n).byte()
    minf = torch.tensor(float('-inf'))
    while True:
        best, index = torch.max(scores_, 0)
        if best.item() <= float('-inf'):
            return boxes[retain], scores[retain], retain
        retain[index] = 1
        collision = (box_overlap(boxes[index], boxes) > 0.5).reshape(-1)
        scores_= torch.where(collision, minf, scores_)    

def topn(boxes, scores, n):
    "Sort the boexes and return the top n"
    n = min(n, len(boxes))
    scores, perm = torch.sort(scores, descending=True)
    perm = perm[:n]
    scores = scores[:n]
    boxes = torch.index_select(boxes, 0, perm)
    return boxes, scores, perm

def collect_hard_negatives(hog_extractor, w, scales, hogs, boxes, labels):
    negs = []
    for index in [i for i, label in enumerate(labels) if label == -1]:
        # Get the next difficult box.
        box = boxes[index]

        # Reconstruct the scale level of this box.
        cs = hog_extractor.cell_size
        mh = w.shape[2]
        mw = w.shape[3]
        scale = (box[2] - box[0]) / (mh * cs)

        # Find the corresponding level.
        diffs = [(scale - s)**2 for s in scales]
        level = diffs.index(min(diffs))

        # Extract the HOG negative patch and save it.
        u0 = int(box[0] / (cs * scale))
        v0 = int(box[1] / (cs * scale))
        negs.append(hogs[level][0, :, v0:v0+mh, u0:u0+mw][None,:])
    return negs

def evaluate_model(imdb, hog_extractor, w, scales, subset='val', collect_negatives=False, use_gpu=None):
    "Evaluate the model by looping over the specivied subset of the image database."
    # Loop over all images in the dataset
    all_labels = []
    all_scores = []
    negs = []
    misses = 0

    if type(subset) is tuple:
        images = imdb[subset[0]]['images'][subset[1]:subset[2]]
        subset = subset[0]
    else:
        images = imdb[subset]['images']

    for t, image in enumerate(images):
        # Load the image
        pil_image = Image.open(image)

        # Pick all the gt boxes in the selected image
        sel = [i for i, box_image in enumerate(imdb[subset]['box_images']) if box_image == image]
        gt_boxes = imdb[subset]['boxes'][sel]

        # Run the detector
        boxes, scores, hogs= hog_extractor.detect_at_multiple_scales(w, scales, pil_image, use_gpu=use_gpu)
        boxes, scores, perm = topn(boxes, scores, 100)
        boxes, scores, _ = nms(boxes, scores)
        
        # Evaluate the detector and plot the results
        results = eval_detections(gt_boxes, boxes)
        all_labels.append(results['labels'])
        all_scores.append(scores)
        misses += results['misses']
        
        # Collect hard negatives if required
        if collect_negatives:
            negs += collect_hard_negatives(hog_extractor, w, scales, hogs, boxes, results['labels'])

        # Compute the per-image AP
        _, _, ap = pr(results['labels'], scores, misses=results['misses'], plot=False)
        print(f"Evaluating on image {t+1:3d} of {len(images):3d} [{image:15s}]: AP: {ap*100:6.1f}%")
        
    return {
        'labels' : torch.cat(all_labels, 0),
        'scores' : torch.cat(all_scores, 0),
        'misses' : misses,
        'negatives' : negs,
    }