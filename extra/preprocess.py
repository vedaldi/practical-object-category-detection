import os
import lab
import torch
import torchvision
import PIL
import random
from matplotlib import pyplot as plt
from PIL import Image

prefix = os.path.join('data', 'tmp', 'TrainIJCNN2013') ;

images = []
boxes = []
labels = []
patches = []

with open(os.path.join(prefix, 'gt.txt')) as f:
    for line in f.readlines():
        fields = line.split(';')
        images.append(os.path.join(prefix, fields[0]))
        boxes.append([int(x) for x in fields[1:5]])
        labels.append(int(fields[5]))
        patch = Image.open(images[-1])
        patch = patch.crop(boxes[-1])
        patch = patch.resize((64, 64), PIL.Image.ANTIALIAS)
        patch = lab.pil_to_torch(patch)
        patches.append(patch)
        base = os.path.splitext(os.path.basename(images[-1]))[0]
        images[-1] = os.path.join('data', 'signs', base + '.jpeg')
        print(f"Added {images[-1]}")

lab.reset_random_seeds()

patches = torch.cat(patches, 0)

# Split images into train and test
train = list(set(images))
random.shuffle(train)
train = train[:400]
train = torch.tensor([x in train for x in images])
val = torch.tensor([not x for x in train])

def get(lst, bools):
    return [lst[i] for i, x in enumerate(bools) if x]

def get_dict(subset):
    return {
        'images' : sorted(list(set(get(images, subset)))),
        'boxes' : torch.tensor(boxes)[subset].float(),
        'box_images' : get(images, subset),
        'box_labels' : torch.tensor(labels)[subset],
        'box_patches' : patches[subset],
    }

imdb = {
    'train' : get_dict(train),
    'val' : get_dict(val),
}

torch.save(imdb, os.path.join('data', 'signs-data.pth'))
        
